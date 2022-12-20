"""
Tensor parallel GPT with ColossalAI
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedSGD
from einops import rearrange
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from xformers.ops.fmha import LowerTriangularMask, memory_efficient_attention

import colossalai.nn as col_nn


@torch.jit.script    # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # key, query, value projections for all heads, but in a batch
        self.c_attn = col_nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.dtype)
        # output projection
        self.c_proj = col_nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        # regularization
        self.flash_attention = config.flash_attention
        if self.flash_attention:
            self.inner_attn = partial(memory_efficient_attention,
                                      attn_bias=LowerTriangularMask(),
                                      p=config.dropout,
                                      scale=math.sqrt(self.n_embd // self.n_head))
        else:
            self.attn_dropout = col_nn.Dropout(config.dropout)
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size, device=config.device,
                                      dtype=torch.bool)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        B, T, C = qkv.size()
        C = C // 3
        head_size = self.n_embd // self.n_head
        n_head = C // head_size
        if self.flash_attention:
            q, k, v = qkv = rearrange(qkv, 'b s (k h d) -> k b s h d', k=3, h=n_head)
            y = self.inner_attn(q, k, v)
            y = rearrange(y, 'b s h d -> b s (h d)')
        else:
            q, k, v = torch.chunk(qkv, 3, dim=2)
            k = k.view(B, T, n_head, head_size).transpose(1, 2)    # (B, nh, T, hs)
            q = q.view(B, T, n_head, head_size).transpose(1, 2)    # (B, nh, T, hs)
            v = v.view(B, T, n_head, head_size).transpose(1, 2)    # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v    # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = col_nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=config.dtype)
        self.config = config
        self.c_proj = col_nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=config.dtype)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        if self.c_fc.bias is not None:
            torch.nn.init.zeros_(self.c_fc.bias)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        if self.c_proj.bias is not None:
            torch.nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.dropout_1 = col_nn.Dropout(config.dropout)
        self.ln_1 = col_nn.LayerNorm(config.n_embd, bias=config.bias, dtype=config.dtype)
        self.mlp = MLP(config)
        self.dropout_2 = col_nn.Dropout(config.dropout)
        self.ln_2 = col_nn.LayerNorm(config.n_embd, bias=config.bias, dtype=config.dtype)

    def _forward(self, x):
        x = self.ln_1(x + self.dropout_1(self.attn(x)))
        x = self.ln_2(x + self.dropout_2(self.mlp(x)))
        return x

    def forward(self, x):
        if self.config.recompute and self.training:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash_attention: bool = False
    recompute: bool = False
    dtype: torch.dtype = None
    device: torch.device = None


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=col_nn.Embedding(config.vocab_size, config.n_embd, dtype=config.dtype),
                wpe=col_nn.Embedding(config.block_size, config.n_embd, dtype=config.dtype),
                drop=col_nn.Dropout(config.dropout),
                ln_f=col_nn.LayerNorm(config.n_embd, bias=config.bias, dtype=config.dtype),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ))

        self.apply(self._init_weights)

        self.lm_head = col_nn.Classifier(config.n_embd,
                                         config.vocab_size,
                                         weight=self.transformer.wte.weight,
                                         bias=False,
                                         dtype=config.dtype)
        self.cross_entropy = col_nn.CrossEntropyLoss()

    def _init_weights(self, module):
        if isinstance(module, (col_nn.Embedding, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)    # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)    # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)    # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = self.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])    # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, algorithm, learning_rate, weight_decay, *args, **kwargs):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, col_nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, col_nn.LayerNorm, torch.nn.Embedding, col_nn.Embedding)
        for mn, m in self.named_modules():
            if 'lm_head' in mn:
                # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
                # will appear in the no_decay and decay sets respectively after the above.
                # In addition, because named_parameters() doesn't return duplicates, it
                # will only return the first occurence, key'd by 'transformer.wte.weight', below.
                # so let's manually skip 'lm_head.weight'. This will include this tensor
                # into optimization via transformer.wte.weight only, and not decayed.
                continue
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn    # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('flat_param'):
                    # fsdp sharded weights
                    if 'attn' in fpn or 'mlp' in fpn:
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        if algorithm == "SGD":
            optim_cls = FusedSGD
        elif algorithm in ["Adam", "AdamW"]:
            optim_cls = FusedAdam
        else:
            optim_cls = getattr(torch.optim, algorithm)
        optimizer = optim_cls(optim_groups, lr=learning_rate, *args, **kwargs)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
