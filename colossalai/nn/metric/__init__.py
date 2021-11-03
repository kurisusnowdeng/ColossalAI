from torch import nn

from ._utils import calc_acc
from .accuracy_2d import Accuracy2D
from .accuracy_3d import Accuracy3D

_parallel_accuracy = {
    '2d': Accuracy2D,
    '3d': Accuracy3D,
}


class Accuracy(nn.Module):
    def __init__(self, tensor_parallel: str = None):
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.acc = calc_acc
        else:
            self.acc = _parallel_accuracy[tensor_parallel]()

    def forward(self, *args):
        return self.acc(*args)
