import argparse
import math
import os
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from resource import RUSAGE_SELF, getrusage
from typing import Iterable, Optional, Union

import apex.amp as amp
import psutil
import torch
import torch.distributed as dist
from col_gpt import GPT, Block, GPTConfig
from datasets import load_from_disk
from torch._six import inf
from torch.distributed import ProcessGroup
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast, default_data_collator

import colossalai
from colossalai.communication import all_reduce
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers
from colossalai.utils.common import _calc_l2_norm

try:
    from colossalai._C import fused_optim
    from colossalai.utils import multi_tensor_applier
except ImportError:
    multi_tensor_applier = None
    fused_optim = None

gpt2_configs = {
    'gpt2-124m': (768, 12, 12),
    'gpt2-355m': (1024, 16, 24),
    'gpt2-1b': (2048, 16, 24),
    'gpt2-3b': (2560, 20, 32),
    'gpt2-4b': (3072, 24, 36),
    'gpt2-6b': (4096, 32, 28),
    'gpt2-8b': (3072, 24, 72),
    'gpt2-10b': (4096, 32, 48),
    'gpt2-13b': (5120, 40, 40),
    'gpt2-16b': (5120, 40, 50),
    'gpt2-20b': (6144, 48, 44),
    'gpt2-24b': (6144, 48, 52),
    'gpt2-32b': (7168, 56, 52),
    'gpt2-66b': (9216, 72, 64),
    'gpt2-92b': (10240, 80, 72),
    'gpt2-175b': (12288, 96, 96),
}

optim_configs = {
    'SGD': dict(momentum=0),
    'Adam': dict(betas=(0.9, 0.95)),
    'AdamW': dict(betas=(0.9, 0.95)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-parallel-mode', '--tp', type=str)
    parser.add_argument('--tensor-parallel-size', '--tp-size', type=int, default=1)
    parser.add_argument('--model', type=str, default='gpt2-355m')
    parser.add_argument('--optim', type=str, default='AdamW')    # SGD or AdamW
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--init-from', type=str, default='scratch')    # 'scratch' or 'resume' or 'gpt2*'
    parser.add_argument('--dtype', type=str, default='float32')    # 'float32'or 'float16' or 'bfloat16'
    parser.add_argument('--O3', action="store_true")    # mixed precision level 3, for experiment purpose
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--global-batch-size', type=int)
    parser.add_argument('--lr', type=float, default=6e-4)    # max learning rate
    parser.add_argument('--min-lr', type=float,
                        default=6e-5)    # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    parser.add_argument('--decay', type=float, default=1e-2)
    parser.add_argument('--no-bias', action="store_true")    # do we use bias inside LayerNorm and Linear layers?
    parser.add_argument('--dropout', type=float, default=0.0)    # for pretraining 0 is good, for finetuning try 0.1+
    parser.add_argument('--clip', type=float, default=1.0)    # clip gradients at this value, or disable if == 0.0
    parser.add_argument('--flash', action="store_true")    # enable flash attention
    parser.add_argument('--fsdp', action="store_true")
    parser.add_argument('--offload', action="store_true")
    parser.add_argument('--recompute', action="store_true")
    parser.add_argument('--max-iters', type=int, default=600000)    # total number of training iterations
    parser.add_argument('--warmup-iters', type=int, default=2000)    # how many steps to warm up for
    parser.add_argument('--lr-decay-iters', type=int)    # should be ~= max_iters per Chinchilla
    parser.add_argument('--eval-only', action="store_true")
    parser.add_argument('--eval-iters', type=int)
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=1)
    return parser.parse_args()


def build_args():
    disable_existing_loggers()

    args = parse_args()

    tp_config = dict()
    if args.tensor_parallel_mode is not None:
        assert args.tensor_parallel_mode in ['1d', '2d', '2.5d', '3d']
        tp_config['parallel'] = dict(tensor=dict(
            mode=args.tensor_parallel_mode,
            size=args.tensor_parallel_size,
        ))

    colossalai.launch_from_torch(config=tp_config)

    args.block_size = 1024

    args.n_embd, args.n_head, args.n_layer = gpt2_configs[args.model]
    args.dropout = 0.0
    args.optim_args = optim_configs[args.optim] if args.optim in optim_configs else {}

    args.decay_lr = True    # whether to decay the learning rate
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters

    # Distributed settings
    args.backend = 'nccl'    # 'nccl', 'gloo', etc.
    # system
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # various inits, derived attributes, I/O setup
    args.distributed = int(os.environ.get('RANK', -1)) != -1    # is this a distributed run?
    if args.distributed:
        args.world_size = gpc.get_world_size(ParallelMode.GLOBAL)
        args.rank = gpc.get_global_rank()
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(f'cuda:{args.local_rank}')
        args.master_process = args.rank == 0    # this process will do logging, checkpointing etc.
        seed_offset = args.rank    # each process gets a different seed
    else:
        # if not distributed, we are running on a single gpu, and one process
        args.master_process = True
        args.world_size = 1
        seed_offset = 0
    if args.tokenizer_path is not None:
        tokenizer = GPT2TokenizerFast(vocab_file=args.tokenizer_path + '/vocab.json',
                                      merges_file=args.tokenizer_path + '/merges.txt')
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    args.vocab_size = math.ceil(tokenizer.vocab_size / args.world_size) * args.world_size

    if args.global_batch_size is None:
        args.global_batch_size = args.batch_size * gpc.get_world_size(ParallelMode.DATA)
    args.gradient_accumulation_steps = args.global_batch_size // (args.batch_size *
                                                                  gpc.get_world_size(ParallelMode.DATA))

    torch.manual_seed(1337 + seed_offset)
    if args.device == 'cuda':
        torch.cuda.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True    # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True    # allow tf32 on cudnn

    if args.O3:
        args.dtype = 'float16'
    args.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    if args.master_process:
        config = vars(args)
        print(f"Configuration = {config}")

    return args


def build_data(args):

    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    dataset = load_from_disk(args.data_path)

    train_data = dataset["train"]
    train_sampler = DistributedSampler(train_data,
                                       shuffle=True,
                                       num_replicas=gpc.get_world_size(ParallelMode.DATA),
                                       rank=gpc.get_local_rank(ParallelMode.DATA))
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              collate_fn=default_data_collator,
                              batch_size=args.batch_size)
    train_iter = cyclic_iter(train_loader)

    val_data = dataset["val"]
    val_sampler = DistributedSampler(val_data,
                                     shuffle=False,
                                     num_replicas=gpc.get_world_size(ParallelMode.DATA),
                                     rank=gpc.get_local_rank(ParallelMode.DATA))
    val_loader = DataLoader(val_data, sampler=val_sampler, collate_fn=default_data_collator, batch_size=args.batch_size)
    val_iter = cyclic_iter(val_loader)
    if args.eval_iters is None:
        args.eval_iters = len(val_loader)

    return train_iter, val_iter


def build_model_optimizer(args, model_args):
    t0 = time.time()
    if args.init_from == 'scratch':
        # init a new model from scratch
        if args.master_process:
            print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif args.init_from == 'resume':
        raise AssertionError("Resuming not supported")
    elif args.init_from.startswith('gpt2'):
        raise AssertionError("From pretrained not supported")
    # crop down the model block size if desired
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # optimizer
    optimizer = model.configure_optimizers(args.optim, args.lr, args.decay, **args.optim_args)

    model = model.to(args.device)

    args.use_apex_amp = False
    if args.dtype == torch.float16 and not args.O3:
        args.use_apex_amp = True
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if gpc.get_world_size(ParallelMode.DATA) > 1:
        if args.fsdp:
            model = FSDP(
                model,
                process_group=gpc.get_group(ParallelMode.DATA),
                device_id=args.local_rank,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=args.offload),
            )
        else:
            model = DDP(model, process_group=gpc.get_group(ParallelMode.DATA), device_ids=[args.local_rank])
    else:
        model = model.to(args.device)

    # report number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    if gpc.is_initialized(ParallelMode.TENSOR):
        n_params *= gpc.get_world_size(ParallelMode.TENSOR)
    if gpc.get_world_size(ParallelMode.DATA) > 1 and args.fsdp:
        n_params *= gpc.get_world_size(ParallelMode.DATA)
    if args.master_process:
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def get_mem_info():
        cpu_mem = psutil.Process(os.getpid()).memory_info()[0] / 1024**2
        max_cpu_mem = getrusage(RUSAGE_SELF).ru_maxrss / 1024
        if args.world_size > 1:
            reduced = torch.tensor([cpu_mem, max_cpu_mem]).to('cuda')
            torch.distributed.all_reduce(reduced)
            cpu_mem, max_cpu_mem = reduced[0].item(), reduced[1].item()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2 if args.device == 'cuda' else None
        max_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if args.device == 'cuda' else None
        return cpu_mem, max_cpu_mem, gpu_mem, max_gpu_mem

    # report memory usage
    cpu_mem, max_cpu_mem, gpu_mem, max_gpu_mem = get_mem_info()
    if args.master_process:
        print(
            f"model initialization: time {(t1-t0)*1000:.2f}ms, memory {cpu_mem:.2f}MB, peak memory {max_cpu_mem:.2f}MB, "
            f"gpu memory {gpu_mem:.2f}MB, peak gpu memory {max_gpu_mem:.2f}MB")

    return model, optimizer


def get_model_flop(args):
    f = args.n_layer * args.n_embd * (6 * args.n_embd + args.block_size)
    f *= 4 * args.global_batch_size * args.block_size
    f = f * 4 if args.recompute else f * 3
    f += 6 * args.global_batch_size * args.block_size * args.n_embd * args.vocab_size
    return f


@contextmanager
def do_sync(model: torch.nn.Module, sync: bool):
    context = nullcontext
    if sync:
        context = getattr(model, "no_sync", context)
    with context():
        yield


def clip_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp: bool = False,
    fsdp_process_group: Optional[ProcessGroup] = None,
):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        enable_cuda_kernels = device.type == 'cuda' and multi_tensor_applier is not None

    if fsdp:
        if fsdp_process_group is None:
            fsdp_process_group = gpc.get_group(ParallelMode.DATA)

    if norm_type == inf:
        total_norm = torch.tensor(max(p.grad.detach().abs().max() for p in parameters))
        total_norm = total_norm.to(dtype=parameters[0].dtype).to(device)
        if gpc.is_initialized(ParallelMode.TENSOR) and gpc.get_world_size(ParallelMode.TENSOR) > 1:
            total_norm = all_reduce(total_norm, ParallelMode.TENSOR, op=torch.distributed.ReduceOp.MAX)
        if fsdp:
            total_norm = dist.all_reduce(total_norm, group=fsdp_process_group, op=torch.distributed.ReduceOp.MAX)
    else:
        if enable_cuda_kernels:
            total_norm = _calc_l2_norm([p.grad for p in parameters])
        else:
            total_norm = torch.linalg.vector_norm(
                torch.stack(
                    [torch.linalg.vector_norm(p.grad.detach(), norm_type, dtype=torch.float32) for p in parameters]),
                norm_type,
            )
            total_norm = total_norm.to(dtype=parameters[0].dtype).to(device)

        total_norm = total_norm**norm_type
        if gpc.is_initialized(ParallelMode.TENSOR) and gpc.get_world_size(ParallelMode.TENSOR) > 1:
            total_norm = all_reduce(total_norm, ParallelMode.TENSOR)
        if fsdp:
            total_norm = dist.all_reduce(total_norm, group=fsdp_process_group)
        total_norm = total_norm**(1.0 / norm_type)

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        if enable_cuda_kernels:
            grads = [p.grad.detach() for p in parameters]
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(fused_optim.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
        else:
            for p in parameters:
                p.grad.detach().mul_(clip_coeff)
    return


args = build_args()

train_iter, val_iter = build_data(args)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init

model_args = dict(n_layer=args.n_layer,
                  n_head=args.n_head,
                  n_embd=args.n_embd,
                  block_size=args.block_size,
                  dropout=args.dropout,
                  vocab_size=args.vocab_size,
                  bias=not args.no_bias,
                  flash_attention=args.flash,
                  recompute=args.recompute,
                  dtype=torch.float16 if args.O3 else None,
                  device=args.device)

model, optimizer = build_model_optimizer(args, model_args)

model_flop = get_model_flop(args)

amp_ctx = nullcontext()
if args.dtype == torch.bfloat16:
    if args.device == "cuda":
        amp_ctx = torch.cuda.amp.autocast(dtype=args.dtype)
    elif args.device == "cpu":
        amp_ctx = torch.cpu.amp.autocast(dtype=args.dtype)


def get_batch(split='train'):
    batch = next(train_iter) if split == 'train' else next(val_iter)
    x, y = batch['input_ids'].to(args.device), batch['labels'].to(args.device)
    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        X, Y = get_batch('val')
        with amp_ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(i):
    # 1) linear warmup for warmup_iters steps
    if i < args.warmup_iters:
        return args.lr * i / args.warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if i > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (i - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))    # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)


# training loop
while True:

    # determine the learning rate for this iteration
    if args.decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0:
        loss = estimate_loss()
        if loss < best_val_loss:
            best_val_loss = loss
        if args.master_process:
            print(f"step {iter_num}: val loss {loss:.4f}, best loss {best_val_loss:.4f}")

    # termination conditions
    if iter_num >= args.max_iters:
        break

    if iter_num == 0 and args.eval_only:
        break

    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step_loss = 0.
    t0 = time.time()

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    optimizer.zero_grad()

    for micro_step in range(args.gradient_accumulation_steps):
        X, Y = get_batch('train')
        with amp_ctx:
            with do_sync(model, micro_step == args.gradient_accumulation_steps - 1):
                logits, loss = model(X, Y)
        loss /= args.gradient_accumulation_steps
        step_loss += loss.item()
        if args.use_apex_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    if args.clip is not None and args.clip != 0:
        clip_grad_norm(amp.master_params(optimizer),
                       args.clip,
                       fsdp=args.fsdp,
                       fsdp_process_group=model.process_group if args.fsdp else None)

    optimizer.step()

    # timing and logging
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if iter_num % args.log_interval == 0 and args.master_process:
        print(f"iter {iter_num}: loss {step_loss:.4f}, time {dt*1000:.2f}ms, "
              f"throughput {args.global_batch_size*args.block_size/dt:.0f} tokens/s, "
              f"tflops {(model_flop/1e12)/(dt*args.world_size):.2f}" +
              (f", memory {torch.cuda.max_memory_allocated()/1024**3:.2f}GB" if args.device == 'cuda' else ""))
    iter_num += 1

gpc.destroy()
