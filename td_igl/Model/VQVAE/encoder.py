import torch
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn import ReLU
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch.nn.utils import weight_norm

from torch_cluster import fps, knn

from td_igl.Method.embed import embed
from td_igl.Model.VQVAE.vision_transformer import VisionTransformer
from td_igl.Model.VQVAE.point_conv import PointConv


class Encoder(nn.Module):
    def __init__(self, N, dim=128, M=2048):
        super().__init__()

        self.embed = Seq(Lin(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))

        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        # self.conv = PointConv(local_nn=Seq(weight_norm(Lin(3+self.embedding_dim, dim))))
        self.conv = PointConv(
            local_nn=Seq(
                weight_norm(Lin(3 + self.embedding_dim, 256)),
                ReLU(True),
                weight_norm(Lin(256, 256)),
            ),
            global_nn=Seq(
                weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))
            ),
        )

        self.transformer = VisionTransformer(
            embed_dim=dim,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.0,
        )

        self.M = M
        self.ratio = N / M
        self.k = 32

    def forward(self, pc):
        # pc: B x N x D
        B, N, D = pc.shape
        assert N == self.M

        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        idx = fps(pos, batch, ratio=self.ratio)  # 0.0625

        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(pos, pos[idx], edge_index, self.basis)
        pos, batch = pos[idx], batch[idx]

        x = x.view(B, -1, x.shape[-1])
        pos = pos.view(B, -1, 3)

        embeddings = embed(pos, self.basis)

        embeddings = self.embed(torch.cat([pos, embeddings], dim=2))

        out = self.transformer(x, embeddings)

        return out, pos
