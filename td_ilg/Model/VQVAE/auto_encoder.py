import torch
import torch.nn as nn

from td_ilg.Config.cfg import _cfg
from td_ilg.Model.VQVAE.encoder import Encoder
from td_ilg.Model.VQVAE.decoder import Decoder
from td_ilg.Model.VQVAE.vector_quantizer2 import VectorQuantizer2


class AutoEncoder(nn.Module):
    def __init__(self, N, K=512, dim=256, M=2048):
        super().__init__()

        self.encoder = Encoder(N=N, dim=dim, M=M)

        self.decoder = Decoder(latent_channel=dim)

        self.codebook = VectorQuantizer2(K, dim)

        self.default_cfg = _cfg()
        return

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def encode(self, x, bins=256):
        B, _, _ = x.shape

        z_e_x, centers = self.encoder(x)  # B x T x C, B x T x 3

        centers_quantized = ((centers + 1) / 2 * (bins - 1)).long()

        z_q_x_st, loss_vq, perplexity, encodings = self.codebook(z_e_x)
        # print(z_q_x_st.shape, loss_vq.item(), perplexity.item(), encodings.shape)
        return z_e_x, z_q_x_st, centers_quantized, loss_vq, perplexity, encodings

    def forward(self, x, points):
        (
            z_e_x,
            z_q_x_st,
            centers_quantized,
            loss_vq,
            perplexity,
            encodings,
        ) = self.encode(x)

        centers = centers_quantized.float() / 255.0 * 2 - 1

        z_q_x = z_q_x_st

        z_q_x_st = z_q_x_st
        B, N, C = z_q_x_st.shape

        logits, sigma = self.decoder(z_q_x_st, centers, points)

        return logits, z_e_x, z_q_x, sigma, loss_vq, perplexity
