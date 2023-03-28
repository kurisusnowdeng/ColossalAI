import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from colossalai.communication import all_gather, all_reduce, reduce_scatter


class ColumnLinear(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input_, weight, bias, parallel_mode, scatter_activation, data_shape):
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.scatter_activation = scatter_activation

        if scatter_activation:
            input_ = all_gather(input_, 0, parallel_mode)
            assert data_shape is not None
            input_ = input_.reshape(*data_shape, -1)

        ctx.save_for_backward(input_, weight)

        output = F.linear(input_, weight, bias)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        grad_input = grad_output.matmul(weight)

        # Asynchronous reduce
        if ctx.scatter_activation:
            grad_input, handle = reduce_scatter(grad_input, 0, ctx.parallel_mode, async_op=True)
        else:
            grad_input, handle = all_reduce(grad_input, ctx.parallel_mode, async_op=True)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        total_input = total_input.reshape(-1, total_input.shape[-1])

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        handle.wait()

        if ctx.scatter_activation:
            grad_input = grad_input.reshape(-1, grad_input.shape[-1])

        return grad_input, grad_weight, grad_bias, None, None, None


def column_linear(
    input_,
    weight,
    bias,
    parallel_mode,
    scatter_activation,
    data_shape,
):
    return ColumnLinear.apply(
        input_,
        weight,
        bias,
        parallel_mode,
        scatter_activation,
        data_shape,
    )


class RowLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input_, weight, bias, parallel_mode, scatter_activation, data_shape, stream_chunk_num):
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.scatter_activation = scatter_activation
        ctx.data_shape = data_shape

        ctx.save_for_backward(input_, weight)

        if stream_chunk_num > 1:
            with torch.no_grad():
                handle_list = []
                outputs = []
                for i in range(stream_chunk_num):
                    output_parallel = F.linear(input_, weight[i])
                    if scatter_activation:
                        output_parallel = output_parallel.reshape(-1, output_parallel.shape[-1])
                        out, handle = reduce_scatter(output_parallel, 0, parallel_mode, async_op=True)
                    else:
                        out, handle = all_reduce(output_parallel, parallel_mode, async_op=True)
                    outputs.append(out)
                    handle_list.append(handle)
                for handle in handle_list:
                    handle.wait()
                output = torch.cat(outputs, dim=-1)
        else:
            output_parallel = F.linear(input_, weight)
            if scatter_activation:
                output_parallel = output_parallel.reshape(-1, output_parallel.shape[-1])
                output = reduce_scatter(output_parallel, 0, parallel_mode)
            else:
                output = all_reduce(output_parallel, parallel_mode)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.scatter_activation:
            grad_output = all_gather(grad_output, 0, ctx.parallel_mode)

        grad_input = grad_output.matmul(weight)
        if ctx.scatter_activation:
            assert ctx.data_shape is not None
            grad_input = grad_input.reshape(*ctx.data_shape, -1)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        total_input = total_input.reshape(-1, total_input.shape[-1])

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None


def row_linear(
    input_,
    weight,
    bias,
    parallel_mode,
    scatter_activation,
    data_shape,
    stream_chunk_num,
):
    return RowLinear.apply(
        input_,
        weight,
        bias,
        parallel_mode,
        scatter_activation,
        data_shape,
        stream_chunk_num,
    )
