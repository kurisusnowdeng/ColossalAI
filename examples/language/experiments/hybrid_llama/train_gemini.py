import argparse
import math
import os
import time

import torch
from datasets import load_from_disk
from gemini_llama import Llama, LlamaConfig
from torch.utils.data import DataLoader, DistributedSampler
from transformers import LlamaTokenizerFast, default_data_collator
from utils import get_mem_info, get_model_flop, llama_configs

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers
from colossalai.utils import get_current_device
from colossalai.zero import GeminiDDP, GeminiOptimizer, LowLevelZeroOptimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero-stage', type=int, choices=(1, 2, 3))
    parser.add_argument('--zero-size', type=int, default=1)
    parser.add_argument('--model', type=str, default='llama-medium')
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--out-path', type=str)
    parser.add_argument('--init-from', type=str, default='scratch')    # 'scratch' or checkpoint path
    parser.add_argument('--dtype', type=str, default='float32')    # 'float32'or 'float16' or 'bfloat16'
    parser.add_argument('--block-size', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)    # max learning rate
    parser.add_argument('--min-lr', type=float,
                        default=3e-5)    # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    parser.add_argument('--decay', type=float, default=1e-2)
    parser.add_argument('--bias', action="store_true")    # do we use bias inside LayerNorm and Linear layers?
    parser.add_argument('--dropout', type=float, default=0.0)    # for pretraining 0 is good, for finetuning try 0.1+
    parser.add_argument('--clip', type=float, default=1.0)    # clip gradients at this value, or disable if == 0.0
    parser.add_argument('--flash', action="store_true")    # enable flash attention?
    parser.add_argument('--offload', type=str, default='static',
                        choices=('static', 'auto'))    # cpu offloading strategy
    parser.add_argument('--recompute', action="store_true")    # enable activation recomputation?
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

    assert args.zero_stage is not None

    colossalai.launch_from_torch(config=dict(parallel=dict(zero=dict(size=args.zero_size))))

    args.n_embd, args.n_fc, args.n_head, args.n_kv_head, args.n_layer = llama_configs[args.model]

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
    else:
        # if not distributed, we are running on a single gpu, and one process
        args.master_process = True
        args.world_size = 1
        args.rank = 0
    if args.tokenizer_path is not None:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained('hf-internal-testing/llama-tokenizer')
    args.vocab_size = math.ceil(tokenizer.vocab_size / args.world_size) * args.world_size

    args.global_batch_size = args.batch_size * gpc.get_world_size(ParallelMode.GLOBAL)

    torch.manual_seed(1337 + args.rank)
    if args.device == 'cuda':
        torch.cuda.manual_seed(1337 + args.rank)
    torch.backends.cuda.matmul.allow_tf32 = True    # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True    # allow tf32 on cudnn

    args.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    # checkpoint file prefices
    args.ckpt_config = 'config.pt'
    args.ckpt_model = 'model.pt'
    args.ckpt_optim = f'optimizer-{args.rank}.pt'

    if args.master_process:
        config = vars(args)
        print(f"Configuration = {config}")

    return args


def build_data(args):

    def cyclic_iter(data: DataLoader):
        epoch = 0
        while True:
            if isinstance(data.sampler, DistributedSampler):
                data.sampler.set_epoch(epoch)
            for x in data:
                yield x
            epoch += 1

    dataset = load_from_disk(args.data_path)

    train_data = dataset['train']
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              collate_fn=default_data_collator,
                              batch_size=args.batch_size,
                              num_workers=4)
    train_iter = cyclic_iter(train_loader)

    val_data = dataset['val']
    val_sampler = DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
                            sampler=val_sampler,
                            collate_fn=default_data_collator,
                            batch_size=args.batch_size,
                            num_workers=4)
    val_iter = cyclic_iter(val_loader)
    if args.eval_iters is None:
        args.eval_iters = len(val_loader)

    return train_iter, val_iter


def build_model_optimizer(args, model_args):
    t0 = time.time()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    lr = args.lr

    if args.master_process:
        print(f"Initializing a new model from {args.init_from}")

    model_state = None
    optim_state = None
    if args.init_from != 'scratch':
        # init a new model from scratch
        assert os.path.exists(args.init_from), "Checkpoint path not found!"
        config = torch.load(os.path.join(args.init_from, args.ckpt_config), map_location='cpu')
        iter_num = config['iter_num']
        best_val_loss = config['best_val_loss']
        lr = config['lr']
        model_state = torch.load(os.path.join(args.init_from, args.ckpt_model), map_location='cpu')
        optim_state = torch.load(os.path.join(args.init_from, args.ckpt_optim), map_location='cpu')

    # build model with gemini
    config = LlamaConfig(**model_args)
    if args.zero_stage == 3:
        config.device = 'meta'
    model = Llama(config)

    # report number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    if args.master_process:
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    if args.zero_stage == 3:
        model = GeminiDDP(model,
                          device_id=get_current_device(),
                          placement_policy=args.offload,
                          mixed_precision=args.dtype,
                          pin_memory=True,
                          search_range_m=128,
                          hidden_dim=model.config.n_embd)
    if model_state is None:
        Llama.init_weights(model)
    else:
        Llama.load_state_dict(model, model_state)

    # build gemini optimizer
    optimizer = Llama.configure_optimizer(model, args.lr, args.decay)
    if args.zero_stage == 3:
        optimizer = GeminiOptimizer(optimizer, model, gpu_margin_mem_ratio=0., max_norm=args.clip)
    else:
        optimizer = LowLevelZeroOptimizer(optimizer,
                                          overlap_communication=True,
                                          forced_dtype=args.dtype,
                                          clip_grad_norm=args.clip,
                                          partition_grad=args.zero_stage == 2,
                                          cpu_offload=args.offload == 'auto')
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    # report memory usage
    cpu_mem, max_cpu_mem, gpu_mem, max_gpu_mem = get_mem_info(args.device == 'cuda')
    if args.master_process:
        print(
            f"model initialization: time {(t1-t0)*1000:.2f}ms, memory {cpu_mem:.2f}MB, peak memory {max_cpu_mem:.2f}MB, "
            f"gpu memory {gpu_mem:.2f}MB, peak gpu memory {max_gpu_mem:.2f}MB")

    return model, optimizer, iter_num, best_val_loss, lr


args = build_args()

train_iter, val_iter = build_data(args)

# model init
model_args = dict(n_layer=args.n_layer,
                  n_kv_head=args.n_kv_head,
                  n_head=args.n_head,
                  n_embd=args.n_embd,
                  n_fc=args.n_fc,
                  block_size=args.block_size,
                  dropout=args.dropout,
                  vocab_size=args.vocab_size,
                  bias=args.bias,
                  flash_attention=args.flash,
                  recompute=args.recompute,
                  device=args.device)

model, optimizer, start_from, best_val_loss, lr = build_model_optimizer(args, model_args)

model_flop = get_model_flop(args)


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


def save_checkpoint(args, model, optimizer, iter_num, **kwargs):
    path = os.path.join(args.out_path, str(iter_num))
    os.makedirs(path, exist_ok=True)
    if args.master_process:
        config = dict(iter_num=iter_num, **kwargs)
        torch.save(config, os.path.join(path, args.ckpt_config))
        torch.save(model.state_dict(), os.path.join(path, args.ckpt_model))
    torch.save(optimizer.state_dict(), os.path.join(path, args.ckpt_optim))


iter_num = 0

# training loop
while True:

    # evaluate the loss on train/val sets and write checkpoints
    if not (iter_num == 0 and args.offload == 'auto') and iter_num % args.eval_interval == 0:
        t0 = time.time()
        loss = estimate_loss()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        if loss < best_val_loss:
            best_val_loss = loss
        if args.master_process:
            print(f"step {iter_num}: {(t1-t0)*1000:.2f}ms, val loss {loss:.4f}, best loss {best_val_loss:.4f}")
        if args.out_path is not None and iter_num > 0:
            save_checkpoint(args,
                            model,
                            optimizer,
                            start_from + iter_num,
                            loss=loss,
                            best_val_loss=best_val_loss,
                            lr=lr)
            if args.master_process:
                print(f"Checkpoint saved to {os.path.join(args.out_path, str(start_from + iter_num))} .")

    # determine the learning rate for this iteration
    if args.decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.lr

    # termination conditions
    if start_from + iter_num >= args.max_iters:
        break

    if iter_num == 0 and args.eval_only:
        break

    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t0 = time.time()

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    optimizer.zero_grad()

    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.backward(loss)

    # scale_before = optimizer.loss_scale
    optimizer.step()
    # scale_after = optimizer.loss_scale
    # overflow = scale_before > scale_after

    # timing and logging
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if iter_num % args.log_interval == 0 and args.master_process:
        print(f"iter {start_from+iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, "
              f"throughput {args.global_batch_size*args.block_size/dt:.0f} tokens/s, "
              f"tflops {(model_flop/1e12)/(dt*args.world_size):.2f}" +
              (f", memory {torch.cuda.max_memory_allocated()/1024**3:.2f}GB" if args.device == 'cuda' else ""))
        # (f", overflowed with scale {scale_before.item() if isinstance(scale_before, torch.Tensor) else scale_before:.5g}" \
        # if overflow else ""))

    iter_num += 1

gpc.destroy()
