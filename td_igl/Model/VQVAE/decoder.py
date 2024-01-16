import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq

from td_igl.Method.embed import embed
from td_igl.Model.VQVAE.embedding import Embedding
from td_igl.Model.VQVAE.vision_transformer import VisionTransformer


class Decoder(nn.Module):
    def __init__(self, latent_channel=192):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        # , nn.GELU(), Lin(128, 128))
        self.embed = Seq(Lin(48 + 3, latent_channel))

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

        self.transformer = VisionTransformer(
            embed_dim=latent_channel,
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

    def forward(self, latents, centers, samples):
        # kernel average
        # samples: B x N x 3
        # latents: B x T x 320
        # centers: B x T x 3

        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        pdist = (
            (samples[:, :, None] - centers[:, None]).square().sum(dim=3)
        )  # B x N x T
        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)

        latents = torch.sum(
            weight[:, :, :, None] * latents[:, None, :, :], dim=2
        )  # B x N x 128
        preds = self.fc(samples, latents).squeeze(2)

        return preds, sigma
