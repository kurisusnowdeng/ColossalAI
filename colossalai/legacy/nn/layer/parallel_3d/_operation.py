from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from colossalai.communication import all_gather, all_reduce, reduce_scatter
from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.kernel import fusion
from colossalai.legacy.communication import all_gather, all_reduce, reduce_scatter

from ._utils import async_grad_bucket as bucket
from ._utils import get_parallel_mode_from_env


class _Linear3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        output_x_weight_parallel_mode: ParallelMode,
        enable_async: bool,
        skip_bias_add: bool,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.skip_bias_add = skip_bias_add
        ctx.enable_async = enable_async
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.output_x_weight_parallel_mode = output_x_weight_parallel_mode

        input_, op_i = all_gather(input_, 0, input_parallel_mode, async_op=True)
        weight, op_w = all_gather(weight, 0, weight_parallel_mode, async_op=True)
        if bias is not None:
            bias = all_gather(bias, -1, output_x_weight_parallel_mode)
        op_i.wait()
        op_w.wait()
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, 0, output_parallel_mode)

        if bias is None:
            return output
        else:
            if skip_bias_add:
                return output, bias
            else:
                return output + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors

        if ctx.use_bias:
            if ctx.skip_bias_add:
                output_grad, bias_grad = output_grads
            else:
                output_grad, = output_grads
                bias_grad = torch.sum(output_grad, dim=tuple(range(output_grad.ndim - 1)))
            bias_grad = reduce_scatter(bias_grad, -1, ctx.output_x_weight_parallel_mode)
        else:
            output_grad, = output_grads
            bias_grad = None

        output_grad = all_gather(output_grad, 0, ctx.output_parallel_mode)

        input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
        weight_grad = torch.matmul(
            input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))

        input_grad, op_i = reduce_scatter(input_grad, 0, ctx.input_parallel_mode, async_op=True)

        if ctx.enable_async:
            bucket.wait()
            if len(bucket) > 1:
                weight_grad = bucket.async_reduce(weight_grad, ctx.weight_parallel_mode, scatter=True)
            else:
                bucket.clear()
                weight_grad = reduce_scatter(weight_grad, 0, ctx.weight_parallel_mode)
        else:
            weight_grad = reduce_scatter(weight_grad, 0, ctx.weight_parallel_mode)
        op_i.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None, None


def linear_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    output_x_weight_parallel_mode: ParallelMode,
    training: bool,
    enable_async: bool,
    skip_bias_add: bool,
) -> Tensor:
    r"""Linear layer for 3D parallelism.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    if training and enable_async and not weight.is_in_bucket:
        bucket.push(weight)
    return _Linear3D.apply(
        input_,
        weight,
        bias,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        output_x_weight_parallel_mode,
        enable_async,
        skip_bias_add,
    )


class _Classifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        input_x_weight_parallel_mode: ParallelMode,
        enable_async: bool,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_x_weight_parallel_mode = input_x_weight_parallel_mode
        ctx.enable_async = enable_async

        weight = weight.transpose(0, 1)
        weight = all_gather(weight, 0, input_x_weight_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = all_reduce(output, output_parallel_mode)

        if bias is not None:
            output += bias

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors

        if ctx.use_bias:
            bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad = all_reduce(bias_grad, ctx.input_x_weight_parallel_mode)
        else:
            bias_grad = None

        weight_grad = torch.matmul(
            input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
        weight_grad, op = reduce_scatter(weight_grad, 0, ctx.input_x_weight_parallel_mode, async_op=True)
        weight_grad = weight_grad.transpose(0, 1)

        input_grad = torch.matmul(output_grad, weight.transpose(0, 1))

        op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    input_x_weight_parallel_mode: ParallelMode,
    enable_async: bool,
) -> Tensor:
    r"""3D parallel classifier.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Classifier3D.apply(
        input_,
        weight,
        bias,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        input_x_weight_parallel_mode,
        enable_async,
    )


class _VocabParallelClassifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
        output_x_weight_parallel_mode: ParallelMode,
        enable_async: bool,
    ) -> Tensor:
        ctx.use_bias = bias is not None

        input_, op_i = all_gather(input_, 0, input_parallel_mode, async_op=True)
        weight, op_w = all_gather(weight, 0, weight_parallel_mode, async_op=True)
        if bias is not None:
            bias = all_gather(bias, -1, output_x_weight_parallel_mode)
        op_w.wait()
        weight = weight.transpose(0, 1)
        op_i.wait()
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, 0, output_parallel_mode)

        if bias is not None:
            output += bias

        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.output_x_weight_parallel_mode = output_x_weight_parallel_mode
        ctx.enable_async = enable_async
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        if ctx.use_bias:
            bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad = reduce_scatter(bias_grad, -1, ctx.output_x_weight_parallel_mode)
        else:
            bias_grad = None

        output_grad = all_gather(output_grad, 0, ctx.output_parallel_mode)

        input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
        weight_grad = torch.matmul(
            output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), input_.reshape(-1, input_.shape[-1]))

        input_grad, op_i = reduce_scatter(input_grad, 0, ctx.input_parallel_mode, async_op=True)
        weight_grad, op_w = reduce_scatter(weight_grad, 0, ctx.weight_parallel_mode, async_op=True)
        op_i.wait()
        op_w.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def vocab_parallel_classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
    output_x_weight_parallel_mode: ParallelMode,
    enable_async: bool,
) -> Tensor:
    r"""3D vocab parallel classifier.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _VocabParallelClassifier3D.apply(
        input_,
        weight,
        bias,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
        output_x_weight_parallel_mode,
        enable_async,
    )


@fusion("nvprims_nvfuser")
def norm_forward(x: Tensor, mean: Tensor, sqr_mean: Tensor, weight: Tensor, bias: Tensor, eps: Tensor):
    mu = x - mean
    var = sqr_mean - mean**2
    sigma = torch.sqrt(var + eps)
    z = mu / sigma
    output = weight * z + bias

    return output, mu, sigma


@fusion("nvprims_nvfuser")
def norm_backward(grad: Tensor, mu: Tensor, sigma: Tensor, weight: Tensor):
    # dbias, dweight = grad, grad * mu / sigma
    dz = grad * weight
    dmu = dz / sigma
    dvar = dz * mu * (-0.5) * sigma**(-3)
    dmean = -dmu
    dvar = torch.sum(dvar, -1, keepdim=True)
    dmean = torch.sum(dmean, -1, keepdim=True)

    return dmu, dmean, dvar


class _Layernorm3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        normalized_shape: int,
        eps: Tensor,
        output_parallel_mode: ParallelMode,
        input_x_weight_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        weight, op_w = all_gather(weight, 0, input_x_weight_parallel_mode, async_op=True)
        if bias is not None:
            bias, op_b = all_gather(bias, 0, input_x_weight_parallel_mode, async_op=True)
        sum_ = torch.sum(input_, dim=-1, keepdim=True)
        sqr_sum = torch.sum(input_**2, dim=-1, keepdim=True)
        mean, sqr_mean = all_reduce(torch.stack((sum_, sqr_sum)), output_parallel_mode) / normalized_shape

        op_w.wait()
        if bias is not None:
            op_b.wait()
        output, mu, sigma = norm_forward(input_, mean, sqr_mean, weight, bias, eps)

        ctx.save_for_backward(mu, sigma, weight)

        ctx.normalized_shape = normalized_shape
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_x_weight_parallel_mode = input_x_weight_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        mu, sigma, weight = ctx.saved_tensors

        weight_grad = output_grad * mu / sigma
        weight_grad = torch.sum(weight_grad, dim=tuple(range(len(weight_grad.shape))[:-1]))
        weight_grad, op_w = reduce_scatter(weight_grad, 0, ctx.input_x_weight_parallel_mode, async_op=True)
        if ctx.use_bias:
            bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad, op_b = reduce_scatter(bias_grad, 0, ctx.input_x_weight_parallel_mode, async_op=True)
        else:
            bias_grad = None

        dmu, dmean, dvar = norm_backward(output_grad, mu, sigma, weight)
        dvar, dmean = all_reduce(torch.stack((dvar, dmean)), ctx.output_parallel_mode)
        input_grad = dmu + (dmean + 2 * dvar * mu) / ctx.normalized_shape

        op_w.wait()
        if ctx.use_bias:
            op_b.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None


def layernorm_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor,
    normalized_shape: int,
    eps: Tensor,
    output_parallel_mode: ParallelMode,
    input_x_weight_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D parallel Layernorm.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.
        input_x_weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input x weight parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Layernorm3D.apply(
        input_,
        weight,
        bias,
        normalized_shape,
        eps,
        output_parallel_mode,
        input_x_weight_parallel_mode,
    )


def split_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""Splits 3D parallel tensor in specified dimension.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): Parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    dim_size = tensor.size(dim)
    world_size = gpc.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, \
        f'The dimension {dim} to split, size ({dim_size}) is not a multiple of world size ({world_size}), ' \
        f'cannot split tensor evenly'
    if tensor.size(dim) <= 1:
        return tensor
    output = torch.chunk(tensor, gpc.get_world_size(parallel_mode),
                         dim=dim)[gpc.get_local_rank(parallel_mode)].contiguous()
    return output


def split_batch_3d(input_: Tensor,
                   dim: int = 0,
                   input_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_INPUT,
                   weight_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_WEIGHT) -> Tensor:
    r"""Splits 3D tensor in batch.

    Args:
        input_ (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): weight parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    if input_.size(dim) <= 1:
        return input_
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_world_size = gpc.get_world_size(weight_parallel_mode)
    input_world_size = gpc.get_world_size(input_parallel_mode)
    output = torch.chunk(input_, weight_world_size, dim=dim)[gpc.get_local_rank(weight_parallel_mode)].contiguous()
    output = torch.chunk(output, input_world_size, dim=dim)[gpc.get_local_rank(input_parallel_mode)].contiguous()
    return output


class _ReduceTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_3d(tensor: Tensor, parallel_mode: ParallelMode) -> Tensor:
    r"""All-reduce the input

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    return _ReduceTensor3D.apply(tensor, parallel_mode)


class _AllGatherTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        output = all_gather(input_, dim, parallel_mode)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        input_grad = reduce_scatter(output_grad, ctx.dim, ctx.parallel_mode)
        return input_grad, None, None


def all_gather_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""All-reduce the gradient in backward pass.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    return _AllGatherTensor3D.apply(tensor, dim, parallel_mode)


class _ReduceScatterTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        return reduce_scatter(input_, dim, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        input_grad = all_gather(output_grad, ctx.dim, ctx.parallel_mode)
        return input_grad, None, None


def reduce_scatter_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""Reduce-scatter the input.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to scatter.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    dim_size = tensor.size(dim)
    world_size = gpc.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, \
        f'The batch size ({dim_size}) is not a multiple of square of 3D depth ({world_size}).'

    return _ReduceScatterTensor3D.apply(tensor, dim, parallel_mode)


class _ReduceByBatch3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                input_: Tensor,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                reduce_mean: bool = False) -> Tensor:
        output = all_reduce(input_, input_parallel_mode)
        output = all_reduce(output, weight_parallel_mode)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = gpc.get_world_size(input_parallel_mode) * gpc.get_world_size(weight_parallel_mode)
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None, None
        else:
            return output_grad, None, None, None


def reduce_by_batch_3d(tensor: Tensor,
                       input_parallel_mode: ParallelMode,
                       weight_parallel_mode: ParallelMode,
                       reduce_mean: bool = False) -> Tensor:
    r"""All-reduce the input from the model parallel region.

    Args:
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        reduce_mean (bool, optional): If set to ``True``, it will divide the output by
            (input parallel size * weight parallel size), default to False.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _ReduceByBatch3D.apply(tensor, input_parallel_mode, weight_parallel_mode, reduce_mean)
