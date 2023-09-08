import os
from resource import RUSAGE_SELF, getrusage

import psutil
import torch

llama_configs = {
    'llama-small': (768, 2048, 12, 12, 12),
    'llama-medium': (1024, 2816, 16, 16, 24),
    'llama-large': (1280, 3584, 20, 20, 36),
    'llama-1b': (2048, 5632, 16, 16, 22),
    'llama-2b': (2560, 6912, 20, 20, 24),
    'llama-3b': (2560, 6912, 20, 20, 32),
    'llama-4b': (3072, 8192, 24, 24, 36),
    'llama-7b': (4096, 11008, 32, 32, 32),
    'llama-10b': (4096, 11008, 32, 32, 48),
    'llama-13b': (5120, 13824, 40, 40, 40),
    'llama-16b': (5120, 13824, 40, 40, 50),
    'llama-20b': (6144, 16384, 48, 48, 44),
    'llama-24b': (6144, 16384, 48, 48, 52),
    'llama-28b': (6656, 17920, 52, 52, 52),
    'llama-33b': (6656, 17920, 52, 52, 60),
    'llama-65b': (8192, 22016, 64, 64, 80),
    'llama-70b': (8192, 28672, 64, 8, 80),
    'llama-92b': (10240, 33536, 80, 10, 72),
    'llama-130b': (12288, 40960, 96, 12, 70),
    'llama-176b': (14336, 47360, 112, 14, 70),
    'llama-280b': (16384, 57344, 128, 32, 80),
}


def get_model_flop(args):
    f = args.n_layer * args.n_embd * (6 * args.n_embd + args.block_size)
    f *= 4 * args.global_batch_size * args.block_size
    f = f * 4 if args.recompute else f * 3
    f += 6 * args.global_batch_size * args.block_size * args.n_embd * args.vocab_size
    return f


def get_mem_info(cuda=True):
    cpu_mem = psutil.Process(os.getpid()).memory_info()[0] / 1024**2
    max_cpu_mem = getrusage(RUSAGE_SELF).ru_maxrss / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024**2 if cuda else None
    max_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if cuda else None
    return cpu_mem, max_cpu_mem, gpu_mem, max_gpu_mem
