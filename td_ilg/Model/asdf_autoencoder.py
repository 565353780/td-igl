import torch
import numpy as np
import torch.nn as nn
from typing import Union

from a_sdf.Model.asdf_model import ASDFModel

from td_ilg.Model.asdf_encoder import ASDFEncoder


class ASDFAutoEncoder(nn.Module):
    def __init__(self, asdf_channel=40, sh_2d_degree=3, sh_3d_degree=6, hidden_dim=128):
        super().__init__()
        self.rad_density = 10

        self.asdf_encoder = ASDFEncoder(
            asdf_channel, sh_2d_degree, sh_3d_degree, hidden_dim
        )

        self.asdf_model = ASDFModel(
            max_sh_3d_degree=sh_3d_degree,
            max_sh_2d_degree=sh_2d_degree,
            dtype=torch.float32,
            device="cpu",
            sample_direction_num=200,
            direction_upscale=4,
        )
        return

    def decodeASDF(self, asdf_params: torch.Tensor) -> torch.Tensor:
        self.asdf_model.loadTorchParams(asdf_params)
        return self.asdf_model.forwardASDF(self.rad_density)

    def forward(
        self, points: torch.Tensor, idxs: Union[np.ndarray, torch.Tensor, None] = None
    ) -> torch.Tensor:
        assert points.shape[0] == 1

        asdf_params = self.asdf_encoder(points, idxs)[0]
        asdf_points = self.decodeASDF(asdf_params).unsqueeze(0)
        return asdf_points
