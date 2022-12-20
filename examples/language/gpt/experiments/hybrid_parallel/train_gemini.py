"""
python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 train_gemini.py \
    --data-path '/data2/users/lcbzd/gpt2_openwebtext'
"""

from functools import partial
from packaging import version
from time import time
import os
import psutil

from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from transformers import GPT2TokenizerFast, default_data_collator
import torch
import torch.nn as nn

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

from gemini_gpt import model_builder


CAI_VERSION = colossalai.__version__
SEQ_LEN = 1024
VOCAB_SIZE = 50257

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
    'AdamW': dict(betas=(0.9, 0.95)),
}


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--shardinit",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument('--model_type', type=str, default='gpt2_medium')
    parser.add_argument('--optim', type=str, default='AdamW')    # SGD or AdamW
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--out-path', type=str)
    parser.add_argument('--init-from', type=str, default='scratch')    # 'scratch' or 'resume' or 'gpt2*'
    parser.add_argument('--dtype', type=str, default='float32')    # 'float32'or 'float16' or 'bfloat16'
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
    parser.add_argument('--max-iters', '--train_step', type=int, default=600000)    # total number of training iterations
    parser.add_argument('--warmup-iters', type=int, default=2000)    # how many steps to warm up for
    parser.add_argument('--lr-decay-iters', type=int)    # should be ~= max_iters per Chinchilla
    parser.add_argument('--eval-only', action="store_true")
    parser.add_argument('--eval-iters', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=1)
    return parser.parse_args()


# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        print(logits)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_gpu_max_mem():
    return torch.cuda.max_memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_batch(data_iter, device):
    batch = next(data_iter)
    x, y = batch['input_ids'].to(device), batch['labels'].to(device)
    return x, y


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.

    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by two modules
            if hasattr(param, 'visited'):
                continue

            # if shard init, then convert param to replica and use the dp-only ProcessGroup
            param: ColoParameter = param
            param.set_dist_spec(ReplicaSpec())
            param.set_process_group(pg)

            # shard it w.r.t tp pattern
            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)    # row slice
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice
            else:
                param.set_dist_spec(ReplicaSpec())
            param.visited = True


def bulid_data_iter(args):
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    assert args.data_path is not None
    dataset = load_from_disk(args.data_path)
    args.use_ddp = args.tp_degree == 1

    train_data = dataset["train"]
    train_sampler = DistributedSampler(train_data, shuffle=True) if args.use_ddp else RandomSampler(train_data)
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              collate_fn=default_data_collator,
                              batch_size=args.batch_size)
    train_iter = cyclic_iter(train_loader)

    val_data = dataset["val"]
    val_sampler = DistributedSampler(val_data, shuffle=False) if args.use_ddp else SequentialSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, collate_fn=default_data_collator, batch_size=args.batch_size)
    val_iter = cyclic_iter(val_loader)
    if args.eval_iters is None:
        args.eval_iters = len(val_loader)

    return train_iter, val_iter


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")
    
    args = parse_args()
    set_cpu_maximum_parallelism()

    # load tokenizers
    if args.tokenizer_path:
        args.tokenizer = GPT2TokenizerFast(vocab_file=args.tokenizer_path + '/vocab.json',
                                      merges_file=args.tokenizer_path + '/merges.txt')
    else:
        args.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    NUM_STEPS = args.max_iters

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, batch size {BATCH_SIZE}", ranks=[0])
    
    args.rank = gpc.get_global_rank()
    args.is_master_process = args.rank == 0

    # build criterion
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    # all param must use the same process group.
    world_size = torch.distributed.get_world_size()
    shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None

    # build GPT model
    with ColoInitContext(device=get_current_device(),
                            dtype=torch.half,
                            default_dist_spec=default_dist_spec,
                            default_pg=shard_pg):
        model = model_builder(args.model_type)(checkpoint=True)

    tp_pg = ProcessGroup(tp_degree=args.tp_degree)
    # Tensor Parallelism (TP)
    # You should notice that v0.1.10 is not compatible with TP degree > 1
    if args.tp_degree > 1:
        tensor_parallelize(model, tp_pg)

    # asign running configurations
    gemini_config = None
    gemini_config = dict(strict_ddp_mode=args.tp_degree == 1,
                            device=get_current_device(),
                            placement_policy=args.placement,
                            pin_memory=True,
                            hidden_dim=model.config.n_embd,
                            search_range_mb=128)
    optim_config = dict(gpu_margin_mem_ratio=0.)

    # build a highly optimized gpu/cpu optimizer
    optimizer = HybridAdam(model.parameters(), lr=1e-4)

    zero_stage = 3
    # wrap your model and optimizer
    model = zero_model_wrapper(model, zero_stage, gemini_config)
    optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)

    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    # model is shared after TP
    numel = get_model_size(model)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
    tflops_list = []
    train_iter, valid_iter = bulid_data_iter(args)

    def train_step():
        model.train()
        start = time()
        optimizer.zero_grad()
        input_ids, labels = get_batch(train_iter, get_current_device())
        outputs = model(input_ids)
        loss = criterion(outputs, labels)    
        optimizer.backward(loss)
        optimizer.step()
        step_time = time() - start
        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss: {loss.item():.3f}, "
            f"Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, "
            f"Peak memory usage: {get_gpu_max_mem():.2f} MB",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)
            
    @torch.no_grad()
    def eval_step():
        model.eval()
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            input_ids, labels = get_batch(valid_iter, get_current_device())
            logits = model(input_ids)
            loss = criterion(logits, labels)
            losses[k] = loss.item()
        out = losses.mean()
        if args.is_master_process:
            logger.info(f"[VAL]: loss {out:.4f}")
        return out

    # best_val_loss = torch.inf
    for n in range(NUM_STEPS):
        train_step()
        if (n + 1) % args.eval_interval == 0:
            loss = eval_step()
            # if loss < best_val_loss and args.out_path is not None:
            #     best_val_loss = loss
            #     if n > 0:
            #         checkpoint = {
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'model_args': model_args,
            #             'iter_num': n,
            #             'best_val_loss': best_val_loss,
            #         }
            #         if args.master_process:
            #             print(f"saving checkpoint to {args.out_path}")
            #             torch.save(checkpoint, os.path.join(args.out_path, 'ckpt.pt'))
            
    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    gpc.destroy()

if __name__ == '__main__':
    main()
