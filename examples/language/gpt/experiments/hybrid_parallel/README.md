# GPT2 benchmark

## PyTorch FSDP
```shell
OMP_NUM_THREADS=128 torchrun --nproc_per_node 4 --master_port 12345 train_fsdp.py --data-path /PATH/TO/PROCESSED/OPENWEBTEXT --tokenizer-path /PATH/TO/GPT2/TOKENIZER --max-iters 10 --eval-iters 1 --warmup-iters 0 --dtype float16 --batch-size 20 --global-batch-size 480 --block-size 512 --model gpt2-10b --optim SGD --recompute --fsdp --O3
```

## ColossalAI Gemini
```shell
cd ../../gemini
DISTPLAN=CAI_Gemini GPUNUM=4 USE_SHARD_INIT=True PLACEMENT=cuda TPDEGREE=1 BATCH_SIZE=20 MODEL_TYPE=gpt2_10b TRAIN_STEP=10 bash run_gemini.sh
```

## Hybrid
