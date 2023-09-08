FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /workspace

RUN pip install -U torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 torchtext torchdata --index-url https://download.pytorch.org/whl/cu118

RUN pip uninstall -y apex && git clone https://github.com/NVIDIA/apex.git && cd apex && \
    python setup.py install --cpp_ext --cuda_ext --fast_layer_norm

RUN pip install -U --no-cache-dir ninja && pip install -v -U --no-cache-dir git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && python setup.py install && cd csrc/rotary && python setup.py install

RUN pip install -U --no-cache-dir transformers datasets

RUN git clone https://github.com/kurisusnowdeng/ColossalAI.git && cd ColossalAI && git checkout hybrid && \
    CUDA_EXT=1 pip install -U -v --no-cache-dir -e .

RUN cd ColossalAI/examples/language/gpt/experiments/hybrid_parallel && \
    wget https://huggingface.co/gpt2/resolve/main/vocab.json && \
    wget https://huggingface.co/gpt2/resolve/main/merges.txt && \
    python process_data.py --name wikitext --block-size 2048 --tokenizer-path ./ --out-path /workspace/data/
