import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from td_igl.Config.gpt import GPTConfig
from td_igl.Model.GPT.block import Block

logger = logging.getLogger(__name__)


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=12,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        n_unmasked=0,
    ):
        super().__init__()
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
        )

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.no_grad()
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("Inf")
        return out

    def forward(self, embeddings):
        x = self.drop(embeddings)
        x = self.blocks(x)

        return x

    def forward_with_past(
        self, idx, embeddings=None, targets=None, past=None, past_length=None
    ):
        # inference only
        assert not self.training
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            # n_layer, 2, b, nh, len_past, dim_head
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [
                self.config.n_layer,
                2,
                idx.shape[0],
                self.config.n_head,
                past_length,
                self.config.n_embd // self.config.n_head,
            ]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            # each position maps to a (learnable) vector
            position_embeddings = self.pos_emb[:, past_length, :]
        else:
            position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(
                x,
                layer_past=past[i, ...] if past is not None else None,
                return_present=True,
            )
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # _, _, n_layer, 2, b, nh, 1, dim_head
        return logits, loss, torch.stack(presents)
