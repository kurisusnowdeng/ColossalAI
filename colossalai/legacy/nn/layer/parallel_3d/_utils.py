from collections import deque
from dataclasses import dataclass
from functools import partial

from torch import Tensor

from colossalai.communication import all_reduce, reduce_scatter
from colossalai.constants import INPUT_GROUP_3D, INPUT_X_WEIGHT_3D, OUTPUT_GROUP_3D, OUTPUT_X_WEIGHT_3D, WEIGHT_GROUP_3D
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env


def get_depth_from_env() -> int:
    try:
        depth = env.depth_3d
        assert depth > 0, 'DEPTH must be greater than zero'
        return depth

    except KeyError:
        raise EnvironmentError('DEPTH is not found in the current environment, '
                               'please make sure that you have used the correct process group initializer')


def is_async_enabled() -> bool:
    try:
        enable_async = env.async_3d
        return enable_async

    except KeyError:
        raise EnvironmentError('3D async mode is not found in the current environment, '
                               'please make sure that you have used the correct process group initializer')


def get_parallel_mode_from_env(group):
    assert group in [INPUT_GROUP_3D, WEIGHT_GROUP_3D, OUTPUT_GROUP_3D, INPUT_X_WEIGHT_3D, OUTPUT_X_WEIGHT_3D], \
        f'{group} is not valid for 3D tensor parallelism.'
    return getattr(env, group)


def swap_in_out_group():
    env.input_group_3d, env.output_group_3d = env.output_group_3d, env.input_group_3d
    env.input_x_weight_group_3d, env.output_x_weight_group_3d = (
        env.output_x_weight_group_3d,
        env.input_x_weight_group_3d,
    )


def dbg_check_shape(tensor: Tensor, shape: tuple):
    rank = gpc.get_global_rank()
    if rank == 0:
        print(tensor.shape)
    assert tensor.shape == shape, \
        '{} does not match {}'.format(tensor.shape, shape)


@dataclass
class AsyncOp:
    tensor: Tensor = None
    grad: Tensor = None
    handle = None


class AsyncGradientBucket(object):

    def __init__(self):
        self.bucket = deque()
        self.maxlen = len(self.bucket)

    def __len__(self):
        return len(self.bucket)

    def clear(self):
        while len(self.bucket) > 0:
            op = self.bucket.pop()
            op.tensor.is_in_bucket = False
        self.maxlen = 0

    def push(self, tensor):
        tensor.is_in_bucket = True
        self.bucket.append(AsyncOp(tensor=tensor))
        self.maxlen = len(self.bucket)

    def async_reduce(self, grad, group, scatter=False, scattered_dim=0):
        assert len(self.bucket) > 0
        op = self.bucket[-1]
        assert op.tensor is not None
        fn = partial(reduce_scatter, dim=scattered_dim) if scatter else all_reduce
        op.grad, op.handle = fn(tensor=grad, parallel_mode=group, async_op=True)
        return None

    def wait(self):
        assert len(self.bucket) > 0
        op = self.bucket[-1]
        if op.handle is not None:
            op.handle.wait()
            if op.tensor.grad is None:
                op.tensor.grad = op.grad
            else:
                op.tensor.grad.add_(op.grad)
            op.tensor.is_in_bucket = False
            self.bucket.pop()


async_grad_bucket = AsyncGradientBucket()
