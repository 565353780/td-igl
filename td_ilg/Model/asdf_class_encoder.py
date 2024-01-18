import torch
import torch.nn as nn
import torch.nn.functional as F

from td_ilg.Model.gpt import GPT
from td_ilg.Method.model import sample
from td_ilg.Config.cfg import _cfg


class ASDFClassEncoder(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        asdf_dim=40,
        ninp=16,
        nhead=2,
        nlayers=24,
        nclasses=55,
        coord_vocab_size=256,
        reso=128,
    ):
        super(ASDFClassEncoder, self).__init__()
        self.reso = reso

        self.pos_emb = nn.Parameter(nn.Embedding(reso, ninp).weight[None])
        self.tpos_emb = nn.Parameter(nn.Embedding(reso, ninp).weight[None])

        self.x_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.y_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.z_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.tx_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.ty_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.tz_tok_emb = nn.Embedding(coord_vocab_size, ninp)
        self.latent_encoder = nn.Linear(asdf_dim - 6, ninp, bias=False)

        self.coord_vocab_size = coord_vocab_size

        self.latent_vocab_size = asdf_dim - 6

        self.class_enc = nn.Embedding(nclasses, ninp)

        self.transformer = GPT(
            vocab_size=512,
            block_size=self.reso,
            n_layer=nlayers,
            n_head=nhead,
            n_embd=ninp,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.ln_x = nn.LayerNorm(ninp)
        self.x_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_y = nn.LayerNorm(ninp)
        self.y_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_z = nn.LayerNorm(ninp)
        self.z_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_tx = nn.LayerNorm(ninp)
        self.tx_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_ty = nn.LayerNorm(ninp)
        self.ty_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_tz = nn.LayerNorm(ninp)
        self.tz_head = nn.Linear(ninp, coord_vocab_size, bias=False)

        self.ln_latent = nn.LayerNorm(ninp)
        self.latent_head = nn.Linear(ninp, asdf_dim - 6, bias=False)

        self.default_cfg = _cfg()
        return

    def forward(self, positions, params, classes):
        features = self.class_enc(classes)[:, None]  # B x 1 x C

        position_embeddings = self.pos_emb  # 1 x S x C
        tposition_embeddings = self.tpos_emb  # 1 x S x C

        x_token_embeddings = self.x_tok_emb(positions[:, :, 0])  # B x S x C
        y_token_embeddings = self.y_tok_emb(positions[:, :, 1])  # B x S x C
        z_token_embeddings = self.z_tok_emb(positions[:, :, 2])  # B x S x C
        tx_token_embeddings = self.tx_tok_emb(positions[:, :, 3])  # B x S x C
        ty_token_embeddings = self.ty_tok_emb(positions[:, :, 4])  # B x S x C
        tz_token_embeddings = self.tz_tok_emb(positions[:, :, 5])  # B x S x C
        latent_features = self.latent_encoder(params)

        token_embeddings = torch.cat(
            [
                features,
                latent_features
                + x_token_embeddings
                + y_token_embeddings
                + z_token_embeddings
                + tx_token_embeddings
                + ty_token_embeddings
                + tz_token_embeddings,
            ],
            dim=1,
        )  # B x (1+S) x C
        embeddings = token_embeddings[:, :-1] + position_embeddings  # B x S x C

        x = self.transformer.drop(embeddings)

        for block in self.transformer.blocks[:12]:
            x = block(x)  # B x S x C
        x_logits = (
            F.log_softmax(self.x_head(self.ln_x(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = x + x_token_embeddings + position_embeddings

        for block in self.transformer.blocks[12:16]:
            x = block(x)
        y_logits = (
            F.log_softmax(self.y_head(self.ln_y(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = x + x_token_embeddings + y_token_embeddings + position_embeddings

        for block in self.transformer.blocks[16:20]:
            x = block(x)
        z_logits = (
            F.log_softmax(self.z_head(self.ln_z(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = (
            x
            + x_token_embeddings
            + y_token_embeddings
            + z_token_embeddings
            + position_embeddings
            + tposition_embeddings
        )

        for block in self.transformer.blocks[20:24]:
            x = block(x)
        tx_logits = (
            F.log_softmax(self.tx_head(self.ln_tx(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = (
            x
            + x_token_embeddings
            + y_token_embeddings
            + z_token_embeddings
            + tx_token_embeddings
            + position_embeddings
            + tposition_embeddings
        )

        for block in self.transformer.blocks[24:28]:
            x = block(x)
        ty_logits = (
            F.log_softmax(self.ty_head(self.ln_ty(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = (
            x
            + x_token_embeddings
            + y_token_embeddings
            + z_token_embeddings
            + tx_token_embeddings
            + ty_token_embeddings
            + position_embeddings
            + tposition_embeddings
        )

        for block in self.transformer.blocks[28:32]:
            x = block(x)
        tz_logits = (
            F.log_softmax(self.tz_head(self.ln_tz(x)), dim=-1)
            .permute(0, 2, 1)
            .view(positions.shape[0], self.coord_vocab_size, self.reso)
        )
        x = (
            x
            + x_token_embeddings
            + y_token_embeddings
            + z_token_embeddings
            + tx_token_embeddings
            + ty_token_embeddings
            + tz_token_embeddings
            + position_embeddings
            + tposition_embeddings
        )

        for block in self.transformer.blocks[32:36]:
            x = block(x)
        latent_logits = self.latent_head(self.ln_latent(x))

        return (
            x_logits,
            y_logits,
            z_logits,
            tx_logits,
            ty_logits,
            tz_logits,
            latent_logits,
        )

    @torch.no_grad()
    def sample(self, cond):
        cond = cond[:, None]

        position_embeddings = self.pos_emb
        tposition_embeddings = self.tpos_emb

        coord1, coord2, coord3, coordt1, coordt2, coordt3, latent = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for _ in range(self.reso):
            if coord1 is None:
                x = self.transformer.drop(cond + position_embeddings[:, :1, :])
                for block in self.transformer.blocks[:12]:
                    x = block(x)  # B x S x C
                coord1_logits = self.x_head(self.ln_x(x))
                ix = sample(coord1_logits)
                coord1 = ix
                x_token_embeddings = self.x_tok_emb(coord1)

                x = x + x_token_embeddings + position_embeddings[:, :1, :]
                for block in self.transformer.blocks[12:16]:
                    x = block(x)  # B x S x C
                coord2_logits = self.y_head(self.ln_y(x))
                ix = sample(coord2_logits)
                coord2 = ix
                y_token_embeddings = self.y_tok_emb(coord2)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + position_embeddings[:, :1, :]
                )
                for block in self.transformer.blocks[16:20]:
                    x = block(x)  # B x S x C
                coord3_logits = self.z_head(self.ln_z(x))
                ix = sample(coord3_logits)
                coord3 = ix
                z_token_embeddings = self.z_tok_emb(coord3)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + position_embeddings[:, :1, :]
                    + tposition_embeddings[:, :1, :]
                )
                for block in self.transformer.blocks[20:24]:
                    x = block(x)  # B x S x C
                coordt1_logits = self.tx_head(self.ln_tx(x))
                ix = sample(coordt1_logits)
                coordt1 = ix
                tx_token_embeddings = self.tx_tok_emb(coordt1)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + position_embeddings[:, :1, :]
                    + tposition_embeddings[:, :1, :]
                )
                for block in self.transformer.blocks[24:28]:
                    x = block(x)  # B x S x C
                coordt2_logits = self.ty_head(self.ln_ty(x))
                ix = sample(coordt2_logits)
                coordt2 = ix
                ty_token_embeddings = self.ty_tok_emb(coordt2)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + ty_token_embeddings
                    + position_embeddings[:, :1, :]
                    + tposition_embeddings[:, :1, :]
                )
                for block in self.transformer.blocks[28:32]:
                    x = block(x)  # B x S x C
                coordt3_logits = self.tz_head(self.ln_tz(x))
                ix = sample(coordt3_logits)
                coordt3 = ix
                tz_token_embeddings = self.tz_tok_emb(coordt3)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + ty_token_embeddings
                    + tz_token_embeddings
                    + position_embeddings[:, :1, :]
                    + tposition_embeddings[:, :1, :]
                )
                for block in self.transformer.blocks[32:36]:
                    x = block(x)  # B x S x C
                latent_logits = self.latent_head(self.ln_latent(x))
                latent = latent_logits

            else:
                x_token_embeddings = self.x_tok_emb(coord1)  # B x S x C
                y_token_embeddings = self.y_tok_emb(coord2)  # B x S x C
                z_token_embeddings = self.z_tok_emb(coord3)  # B x S x C
                tx_token_embeddings = self.x_tok_emb(coordt1)  # B x S x C
                ty_token_embeddings = self.y_tok_emb(coordt2)  # B x S x C
                tz_token_embeddings = self.z_tok_emb(coordt3)  # B x S x C
                latent_features = self.latent_encoder(latent)

                token_embeddings = torch.cat(
                    [
                        cond,
                        latent_features
                        + x_token_embeddings
                        + y_token_embeddings
                        + z_token_embeddings
                        + tx_token_embeddings
                        + ty_token_embeddings
                        + tz_token_embeddings,
                    ],
                    dim=1,
                )  # B x (1+S) x C

                embeddings = (
                    token_embeddings
                    + position_embeddings[:, : token_embeddings.shape[1], :]
                )  # B x S x C

                x = self.transformer.drop(embeddings)
                for block in self.transformer.blocks[:12]:
                    x = block(x)  # B x S x C
                coord1_logits = self.x_head(self.ln_x(x))
                ix = sample(coord1_logits)
                coord1 = torch.cat((coord1, ix), dim=1)
                x_token_embeddings = self.x_tok_emb(coord1)

                x = x + x_token_embeddings + position_embeddings[:, : x.shape[1], :]
                for block in self.transformer.blocks[12:16]:
                    x = block(x)  # B x S x C
                coord2_logits = self.y_head(self.ln_y(x))
                ix = sample(coord2_logits)
                coord2 = torch.cat((coord2, ix), dim=1)
                y_token_embeddings = self.y_tok_emb(coord2)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + position_embeddings[:, : x.shape[1], :]
                )
                for block in self.transformer.blocks[16:20]:
                    x = block(x)  # B x S x C
                coord3_logits = self.z_head(self.ln_z(x))
                ix = sample(coord3_logits)
                coord3 = torch.cat((coord3, ix), dim=1)
                z_token_embeddings = self.z_tok_emb(coord3)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + position_embeddings[:, : x.shape[1], :]
                    + tposition_embeddings[:, : x.shape[1], :]
                )
                for block in self.transformer.blocks[20:24]:
                    x = block(x)  # B x S x C
                coordt1_logits = self.tx_head(self.ln_tx(x))
                ix = sample(coordt1_logits)
                coordt1 = torch.cat((coordt1, ix), dim=1)
                tx_token_embeddings = self.tx_tok_emb(coordt1)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + position_embeddings[:, : x.shape[1], :]
                    + tposition_embeddings[:, : x.shape[1], :]
                )
                for block in self.transformer.blocks[24:28]:
                    x = block(x)  # B x S x C
                coordt2_logits = self.ty_head(self.ln_ty(x))
                ix = sample(coordt2_logits)
                coordt2 = torch.cat((coordt2, ix), dim=1)
                ty_token_embeddings = self.ty_tok_emb(coordt2)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + ty_token_embeddings
                    + position_embeddings[:, : x.shape[1], :]
                    + tposition_embeddings[:, : x.shape[1], :]
                )
                for block in self.transformer.blocks[28:32]:
                    x = block(x)  # B x S x C
                coordt3_logits = self.tz_head(self.ln_tz(x))
                ix = sample(coordt3_logits)
                coordt3 = torch.cat((coordt3, ix), dim=1)
                tz_token_embeddings = self.tz_tok_emb(coordt3)

                x = (
                    x
                    + x_token_embeddings
                    + y_token_embeddings
                    + z_token_embeddings
                    + tx_token_embeddings
                    + ty_token_embeddings
                    + tz_token_embeddings
                    + position_embeddings[:, : x.shape[1], :]
                    + tposition_embeddings[:, : x.shape[1], :]
                )
                for block in self.transformer.blocks[32:36]:
                    x = block(x)  # B x S x C
                ix = self.latent_head(self.ln_latent(x))
                # latent = torch.cat((latent, ix), dim=1)
                latent = ix

        # return coord1, coord2, coord3, coordt1, coordt2, coordt3, latent
        coords = torch.cat(
            [
                coord1.unsqueeze(-1),
                coord2.unsqueeze(-1),
                coord3.unsqueeze(-1),
                coordt1.unsqueeze(-1),
                coordt2.unsqueeze(-1),
                coordt3.unsqueeze(-1),
            ],
            dim=-1,
        )

        return torch.cat(
            [
                coords / 128.0 - 1.0,
                latent,
            ],
            dim=-1,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb", "tpos_emb", "xyz_emb"}
