# GPT2 benchmark

## Preparation

### Dependencies

Install apex
```shell
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cpp_ext --cuda_ext --fast_layer_norm
```

Install xformers
```shell
pip install ninja
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -v -U .
```

Install bitsandbytes (e.g. CUDA 11.8)
```shell
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=118 make cuda11x
python setup.py install
```

Install ColossalAI
```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
CUDA_EXT=1 pip install -v -U .
```

### Dataset

```shell
pip install -U transformers datasets
python process_data.py --out-path /PATH/TO/PROCESSED/OPENWEBTEXT
```

## Usage

### PyTorch FSDP

```shell
OMP_NUM_THREADS=128 torchrun --nproc_per_node 8 --master_port 23333 train_torch.py \
    --data-path /PATH/TO/PROCESSED/OPENWEBTEXT \
    --model gpt2-10b \
    --max-iters 10 --eval-iters 1 --warmup-iters 0 \
    --batch-size 4 --global-batch-size 128 \
    --optim AdamW \
    --dtype float16 \
    --recompute \
    --zero-stage 3
```

### ColossalAI Gemini

```shell
OMP_NUM_THREADS=128 torchrun --nproc_per_node 8 --master_port 23333 train_gemini.py \
    --data-path /PATH/TO/PROCESSED/OPENWEBTEXT \
    --model gpt2-10b \
    --max-iters 10 --eval-iters 1 --warmup-iters 0 \
    --batch-size 4 \
    --optim AdamW \
    --dtype float16 \
    --recompute \
    --flash \
    --zero-stage 3
```

### ColossalAI Tensor Parallelism

```shell
OMP_NUM_THREADS=128 torchrun --nproc_per_node 8 --master_port 23333 train_col.py \
    --data-path /PATH/TO/PROCESSED/OPENWEBTEXT \
    --model gpt2-10b \
    --max-iters 10 --eval-iters 1 --warmup-iters 0 \
    --batch-size 4 --global-batch-size 128 \
    --optim AdamW \
    --dtype float16 --amp-level 2 \
    --recompute \
    --flash \
    --tp 1d --tp-size 4 \
    --zero-stage 3
```
