import torch
import numpy as np
import torch.nn as nn
from math import ceil
from typing import Union
from functools import partial
from torch.nn import ReLU
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch.nn.utils.parametrizations import weight_norm

from torch_cluster import fps, knn

from td_ilg.Method.embed import embed
from td_ilg.Model.VQVAE.vision_transformer import VisionTransformer
from td_ilg.Model.VQVAE.point_conv import PointConv


class ASDFEncoder(nn.Module):
    def __init__(self, asdf_channel=40, sh_2d_degree=3, sh_3d_degree=6, hidden_dim=128):
        super().__init__()
        self.embedding_dim = 48

        self.asdf_channel = asdf_channel
        self.sh_2d_dim = sh_2d_degree * 2 + 1
        self.sh_3d_dim = (sh_3d_degree + 1) ** 2

        self.embed = Lin(self.embedding_dim + 3, hidden_dim)

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

        self.conv = PointConv(
            local_nn=Seq(
                weight_norm(Lin(3 + self.embedding_dim, hidden_dim // 2)),
                ReLU(True),
                weight_norm(Lin(hidden_dim // 2, hidden_dim // 2)),
                ReLU(True),
                weight_norm(Lin(hidden_dim // 2, hidden_dim)),
                ReLU(True),
                weight_norm(Lin(hidden_dim, hidden_dim)),
            ),
            global_nn=Seq(
                weight_norm(Lin(hidden_dim, hidden_dim)),
                ReLU(True),
                weight_norm(Lin(hidden_dim, hidden_dim)),
                ReLU(True),
                weight_norm(Lin(hidden_dim, hidden_dim)),
            ),
        )

        self.xyz_transformer = VisionTransformer(
            embed_dim=hidden_dim,
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
        self.txyz_transformer = VisionTransformer(
            embed_dim=hidden_dim,
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
        self.sh2d_transformer = VisionTransformer(
            embed_dim=hidden_dim,
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
        self.sh3d_transformer = VisionTransformer(
            embed_dim=hidden_dim,
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


        self.ln_xyz = nn.LayerNorm(hidden_dim)
        self.xyz_head = Seq(Lin(hidden_dim, hidden_dim),
                            ReLU(True),
                            Lin(hidden_dim, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, 3))

        self.ln_txyz = nn.LayerNorm(hidden_dim)
        self.txyz_head = Seq(Lin(hidden_dim, hidden_dim),
                            ReLU(True),
                            Lin(hidden_dim, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, 3))

        self.ln_sh2d = nn.LayerNorm(hidden_dim)
        self.sh2d_head = Seq(Lin(hidden_dim, hidden_dim),
                            ReLU(True),
                            Lin(hidden_dim, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, self.sh_2d_dim))

        self.sh2d_embed = Lin(self.sh_2d_dim, hidden_dim)

        self.ln_sh3d = nn.LayerNorm(hidden_dim)
        self.sh3d_head = Seq(Lin(hidden_dim, hidden_dim),
                            ReLU(True),
                            Lin(hidden_dim, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, hidden_dim // 2),
                            ReLU(True),
                            Lin(hidden_dim // 2, self.sh_3d_dim))

        return

    def forward(self, pc, idx: Union[np.ndarray, torch.Tensor, None] = None):
        B, N, D = pc.shape

        pos = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        if idx is None:
            idx = fps(pos, batch, ratio=self.asdf_channel / N)
        else:
            assert idx.shape[1] == self.asdf_channel
            idx = torch.cat([idx[i] + i * N for i in range(idx.shape[0])])

        row, col = knn(pos, pos[idx], ceil(N / self.asdf_channel), batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        points_feature = self.conv(pos, pos[idx], edge_index, self.basis)
        center = pos[idx]

        points_feature = points_feature.view(B, -1, points_feature.shape[-1])
        center = center.view(B, -1, 3)

        center_embeddings = embed(center, self.basis)
        center_embeddings = self.embed(torch.cat([center, center_embeddings], dim=2))

        xyz_feature = self.xyz_transformer(points_feature, center_embeddings)

        delta_xyz = self.xyz_head(self.ln_xyz(xyz_feature))

        delta_xyz_embeddings = embed(delta_xyz, self.basis)
        delta_xyz_embeddings = self.embed(
            torch.cat([delta_xyz, delta_xyz_embeddings], dim=2)
        )

        txyz_feature = self.txyz_transformer(points_feature, center_embeddings + delta_xyz_embeddings)

        delta_txyz = self.txyz_head(self.ln_txyz(txyz_feature))

        delta_txyz_embeddings = embed(delta_txyz, self.basis)
        delta_txyz_embeddings = self.embed(
            torch.cat([delta_txyz, delta_txyz_embeddings], dim=2)
        )

        sh2d_feature = self.sh2d_transformer(
            points_feature, center_embeddings + delta_xyz_embeddings + delta_txyz_embeddings
        )

        sh_2d = self.sh2d_head(self.ln_sh2d(sh2d_feature))

        sh2d_embeddings = self.sh2d_embed(sh_2d)

        sh3d_feature = self.sh3d_transformer(
            points_feature,
            center_embeddings
            + delta_xyz_embeddings
            + delta_txyz_embeddings
            + sh2d_embeddings,
        )

        sh_3d = self.sh3d_head(self.ln_sh3d(sh3d_feature))

        return torch.cat(
            [center + delta_xyz, center + delta_txyz, sh_2d, sh_3d], dim=-1
        )
