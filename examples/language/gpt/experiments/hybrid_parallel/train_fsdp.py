"""
Adapted from nanoGPT:
https://github.com/karpathy/nanoGPT/blob/master/train.py

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with DDP or FSDP.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from resource import RUSAGE_SELF, getrusage

import psutil
import torch
from datasets import load_from_disk
from min_gpt import GPT, GPTConfig
from torch.cuda.amp.grad_scaler import GradScaler as TorchGradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from transformers import GPT2TokenizerFast, default_data_collator

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
    parser.add_argument('--model', type=str, default='gpt2-355m')
    parser.add_argument('--optim', type=str, default='AdamW')    # SGD, Adam or AdamW
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--out-path', type=str)
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
    args = parse_args()

    args.block_size = 1024

    args.n_embd, args.n_head, args.n_layer = gpt2_configs[args.model]
    args.dropout = 0.0
    args.optim_args = optim_configs[args.optim] if args.optim in optim_configs else {}

    args.decay_lr = True    # whether to decay the learning rate
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters

    # DDP settings
    args.backend = 'nccl'    # 'nccl', 'gloo', etc.
    # system
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # various inits, derived attributes, I/O setup
    args.ddp = int(os.environ.get('RANK', -1)) != -1    # is this a ddp run?
    if args.ddp:
        init_process_group(backend=args.backend)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(f'cuda:{args.local_rank}')
        args.master_process = args.rank == 0    # this process will do logging, checkpointing etc.
        seed_offset = args.rank    # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
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
        args.global_batch_size = args.batch_size * args.world_size
    args.gradient_accumulation_steps = args.global_batch_size // (args.batch_size * args.world_size)

    if args.out_path is not None and args.master_process:
        os.makedirs(args.out_path, exist_ok=True)

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
    train_sampler = DistributedSampler(train_data, shuffle=True) if args.ddp else RandomSampler(train_data)
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              collate_fn=default_data_collator,
                              batch_size=args.batch_size)
    train_iter = cyclic_iter(train_loader)

    val_data = dataset["val"]
    val_sampler = DistributedSampler(val_data, shuffle=False) if args.ddp else SequentialSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, collate_fn=default_data_collator, batch_size=args.batch_size)
    val_iter = cyclic_iter(val_loader)
    if args.eval_iters is None:
        args.eval_iters = len(val_loader)

    return train_iter, val_iter


def build_model_optimizer(args, model_args):
    t0 = time.time()
    checkpoint = None
    if args.init_from == 'scratch':
        # init a new model from scratch
        if args.master_process:
            print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif args.init_from == 'resume':
        if args.master_process:
            print(f"Resuming training from {args.out_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_path, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model_args = checkpoint['model_args']
        for k, v in model_args.items():
            assert checkpoint_model_args[k] == v, "for now"
            # TODO: think through how passed in params should interact with checkpoint params
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        args.iter_num = checkpoint['iter_num']
        args.best_val_loss = checkpoint['best_val_loss']
    elif args.init_from.startswith('gpt2'):
        if args.master_process:
            print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        assert not args.no_bias, "GPT-2 models must have bias"
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=args.dropout)
        model = GPT.from_pretrained(args.init_from, override_args)
        # read off and override the GPT sizing model args from the model config
        model_args['n_layer'] = model.config.n_layer
        model_args['n_head'] = model.config.n_head
        model_args['n_embd'] = model.config.n_embd
    # crop down the model block size if desired
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # report number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    if args.fsdp:
        n_params *= args.world_size
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

    # optimizer
    optimizer = model.configure_optimizers(args.optim, args.lr, args.decay, **args.optim_args)
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def get_model_flop(args):
    f = args.n_layer * args.n_embd * (6 * args.n_embd + args.block_size)
    f *= 4 * args.global_batch_size * args.block_size
    f = f * 4 if args.recompute else f * 3
    f += 6 * args.global_batch_size * args.block_size * args.n_embd * args.vocab_size
    return f


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
                  fsdp=args.fsdp,
                  offload=args.offload,
                  recompute=args.recompute,
                  dtype=args.dtype,
                  O3=args.O3,
                  device=args.device,
                  device_id=args.local_rank)

model, optimizer = build_model_optimizer(args, model_args)

model_flop = get_model_flop(args)

amp_ctx = nullcontext() if args.device == 'cpu' else torch.amp.autocast(device_type=args.device, dtype=args.dtype)


class GradScaler(TorchGradScaler):

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ):
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


scaler = None
if args.dtype == torch.float16:
    if args.fsdp:
        scaler = ShardedGradScaler()
    else:
        scaler = GradScaler()


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
            logits, loss = model(X, Y)
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
        if args.master_process:
            print(f"step {iter_num}: val loss {loss:.4f}")
        if loss < best_val_loss and args.out_path is not None:
            best_val_loss = loss
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                if args.master_process:
                    print(f"saving checkpoint to {args.out_path}")
                    torch.save(checkpoint, os.path.join(args.out_path, 'ckpt.pt'))

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
    with amp_ctx:
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_batch('train')
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync(micro_step == args.gradient_accumulation_steps - 1)
            logits, loss = model(X, Y)
            step_loss += loss.item()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if args.clip is not None and args.clip != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if hasattr(model, "clip_grad_norm_"):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                model.clip_grad_norm_(args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        overflow = False
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            overflow = scale_before > scale_after
        else:
            optimizer.step()

    # timing and logging
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if iter_num % args.log_interval == 0 and args.master_process:
        step_loss /= args.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {step_loss:.4f}, time {dt*1000:.2f}ms, "
              f"throughput {args.global_batch_size*args.block_size/dt:.0f} tokens/s, "
              f"tflops {(model_flop/1e12)/(dt*args.world_size):.2f}" +
              (f", memory {torch.cuda.max_memory_allocated()/1024**3:.2f}GB" if args.device == 'cuda' else "") +
              (f", overflowed with scale {scale_before:.5g}" if overflow else ""))
    iter_num += 1

if args.ddp:
    destroy_process_group()
