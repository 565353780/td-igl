import torch
import numpy as np
import torch.nn as nn
from typing import Union

from a_sdf.Model.asdf_model import ASDFModel

from td_ilg.Model.asdf_encoder import ASDFEncoder


class ASDFAutoEncoder(nn.Module):
    def __init__(self, asdf_channel=40, sh_2d_degree=3, sh_3d_degree=6, hidden_dim=128, dtype=torch.float32, device: str='cpu', sample_direction_num: int=200, direction_upscale: int=4):
        super().__init__()
        self.rad_density = 10

        self.asdf_encoder = ASDFEncoder(
            asdf_channel, sh_2d_degree, sh_3d_degree, hidden_dim
        )

        self.asdf_model = ASDFModel(
            max_sh_3d_degree=sh_3d_degree,
            max_sh_2d_degree=sh_2d_degree,
            dtype=dtype,
            device=device,
            sample_direction_num=sample_direction_num,
            direction_upscale=direction_upscale,
        )
        return

    def encodeASDF(
        self, points: torch.Tensor, idxs: Union[np.ndarray, torch.Tensor, None] = None
    ) -> torch.Tensor:
        return self.asdf_encoder(points, idxs)

    def decodeASDF(self, asdf_params: torch.Tensor) -> torch.Tensor:
        self.asdf_model.loadTorchParams(asdf_params)
        return self.asdf_model.forwardASDF(self.rad_density)

    def forward(
        self, points: torch.Tensor, idxs: Union[np.ndarray, torch.Tensor, None] = None
    ) -> torch.Tensor:
        # TODO: allow multi batch for ASDFModel later for faster training speed
        assert points.shape[0] == 1

        asdf_params = self.encodeASDF(points, idxs)
        asdf_points = self.decodeASDF(asdf_params[0]).unsqueeze(0)
        return asdf_points

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}
