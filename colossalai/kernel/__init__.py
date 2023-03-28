from typing import Callable, Union

import torch

from .cuda_native import FusedScaleMaskSoftmax, LayerNorm, MultiHeadAttention
from .triton import llama_context_attn_fwd, bloom_context_attn_fwd
from .triton import softmax
from .triton import copy_kv_cache_to_dest

__all__ = [
    "LayerNorm",
    "FusedScaleMaskSoftmax",
    "MultiHeadAttention",
    "llama_context_attn_fwd",
    "bloom_context_attn_fwd",
    "softmax",
    "copy_kv_cache_to_dest",
    "fusion",
]


def fusion(backend):

    def optimize_by_dynamo(fn: Union[Callable, torch.nn.Module]):
        import torch._dynamo
        torch._dynamo.reset()
        return torch._dynamo.optimize(backend=backend)(fn)

    def optimize_by_jit(fn: Union[Callable, torch.nn.Module]):
        from .jit import set_jit_fusion_options
        set_jit_fusion_options()
        return torch.jit.script(fn)

    if backend == "jit":
        return optimize_by_jit
    else:
        return optimize_by_dynamo
