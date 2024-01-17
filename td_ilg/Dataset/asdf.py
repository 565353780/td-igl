import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from td_ilg.Config.shapenet import CATEGORY_IDS


class ASDFDataset(Dataset):
    def __init__(self, asdf_dataset_folder_path: str) -> None:
        self.asdf_file_list = []
        self.context_files_list = []

        """
        self.asdf_file_list = [1] * 10000
        self.context_files_list = [1] * 10000
        return
        """

        self.loadDataset(asdf_dataset_folder_path)
        return

    def loadDataset(self, asdf_dataset_folder_path: str) -> bool:
        class_foldername_list = os.listdir(asdf_dataset_folder_path)

        for class_foldername in class_foldername_list:
            model_folder_path = asdf_dataset_folder_path + class_foldername + "/"
            if not os.path.exists(model_folder_path):
                continue

            model_filename_list = os.listdir(model_folder_path)

            for model_filename in tqdm(model_filename_list):
                asdf_folder_path = model_folder_path + model_filename + "/"
                if not os.path.exists(asdf_folder_path):
                    continue

                asdf_filename_list = os.listdir(asdf_folder_path)

                if "final.npy" not in asdf_filename_list:
                    continue

                context_files = []

                for asdf_filename in asdf_filename_list:
                    if asdf_filename == "final.npy":
                        continue

                    if asdf_filename[-4:] != ".npy":
                        continue

                    context_files.append(asdf_folder_path + asdf_filename)

                self.asdf_file_list.append(asdf_folder_path + "final.npy")
                self.context_files_list.append(context_files)

                self.asdf_file_list = self.asdf_file_list * 1000000
                self.context_files_list = [self.context_files_list[0]] * 1000000
                return True

        return True

    def __len__(self):
        assert len(self.asdf_file_list) == len(
            self.context_files_list
        ), "Number of feature files and label files should be same"
        return len(self.asdf_file_list)

    def __getitem__(self, idx):
        """
        return (
            torch.randint(0, 256, [100, 6]).type(torch.long),
            torch.rand(100, 34).type(torch.float32),
            CATEGORY_IDS["02691156"],
        )
        """

        asdf_file_path = self.asdf_file_list[idx]
        asdf = np.load(asdf_file_path, allow_pickle=True).item()["params"]

        """
        context_file_path = choice(self.context_files_list[idx])
        context = (
            np.load(context_file_path, allow_pickle=True)
            .item()["params"]
            .reshape(1, 100, 40)
        )
        """

        positions = asdf[:, :6]
        params = asdf[:, 6:]

        embedding_positions = ((positions + 0.5) * 255.0).astype(np.longlong)

        return (
            torch.from_numpy(embedding_positions).type(torch.long),
            torch.from_numpy(params).type(torch.float32),
            CATEGORY_IDS["02691156"],
        )
