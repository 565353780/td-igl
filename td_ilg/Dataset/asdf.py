import os
import torch
import numpy as np
from torch.utils import data

from td_ilg.Config.shapenet import CATEGORY_IDS


class ASDFDataset(data.Dataset):
    def __init__(
        self,
        dataset_folder,
        split,
        categories=None,
        transform=None,
        sampling=True,
        num_samples=4096,
        return_surface=True,
        surface_sampling=True,
        pc_size=2048,
    ):
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, "ShapeNetV2_point")
        self.mesh_folder = os.path.join(self.dataset_folder, "ShapeNetV2_watertight")

        self.models = [
            {
                "category": 0,
                "model": "test",
            }
        ] * 1000
        return

    def __getitem__(self, idx):
        resolution = 12
        coord_vocab_size = 256

        positions = np.random.randint(0, coord_vocab_size, [resolution, 6])
        params = np.random.rand(resolution, 34)
        return (
            torch.from_numpy(positions).type(torch.long),
            torch.from_numpy(params).type(torch.float32),
            CATEGORY_IDS["02691156"],
        )

    def __len__(self):
        return len(self.models)
