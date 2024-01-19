import os
import torch
import numpy as np
from tqdm import tqdm
from random import sample
from torch.utils.data import Dataset


class PointsDataset(Dataset):
    def __init__(
        self,
        points_dataset_folder_path: str,
    ) -> None:
        self.points_file_list = []
        self.min_points_percent = 0.1
        self.max_points_percent = 1.0

        self.loadShapeNetDataset(points_dataset_folder_path)
        return

    def loadShapeNetDataset(self, shapenet_dataset_folder_path: str) -> bool:
        class_foldername_list = os.listdir(shapenet_dataset_folder_path)

        for class_foldername in class_foldername_list:
            points_folder_path = shapenet_dataset_folder_path + class_foldername + "/"
            if not os.path.exists(points_folder_path):
                continue

            # FIXME: only chair here
            if class_foldername != "03001627":
                continue

            points_filename_list = os.listdir(points_folder_path)

            for points_filename in tqdm(points_filename_list):
                if points_filename[-4:] != ".npy":
                    continue

                self.points_file_list.append(points_folder_path + points_filename)

        return True

    def __len__(self):
        return len(self.points_file_list)

    def __getitem__(self, idx):
        points_file_path = self.points_file_list[idx]
        points = np.load(points_file_path, allow_pickle=True)
        shuffle_points = np.random.permutation(points)

        sample_point_num = np.random.randint(int(self.min_points_percent * points.shape[0]), int(self.max_points_percent * points.shape[0]))
        random_idxs = np.array(sample(list(range(points.shape[0])), sample_point_num), dtype=int)
        sample_points = points[random_idxs]

        return torch.from_numpy(sample_points).type(torch.float32), torch.from_numpy(shuffle_points).type(torch.float32)
