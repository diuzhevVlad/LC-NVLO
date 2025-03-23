import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class KittiVLOdomDataset(Dataset):
    """KITTI visual lidar odometry dataset."""

    def __init__(
        self,
        root_dir: str,
        calib_filename: str = "calib.txt",
        diff_gt_poses_filename="diff_gt_poses.npy",
        diff_kiss_poses_filename="diff_kiss_poses.npy",
    ):
        """
        Arguments:
            root_dir (string): Root KITTI directory (dataset).
        """
        self.root_dir = root_dir

        # Initialize sequences information
        seq_path = os.path.join(root_dir, "sequences")
        seq_names = os.listdir(seq_path)
        self.seqs_info = {
            seq: {
                prop: None
                for prop in [
                    "len",
                    "pcd_path",
                    "img0_path",
                    "img1_path",
                    "calib_path",
                    "diff_gt_poses_path",
                    "diff_kiss_poses_path",
                ]
            }
            for seq in seq_names
        }

        # Initialize sequences information
        for _seq_name in seq_names:
            self.seqs_info[_seq_name]["pcd_path"] = os.path.join(
                seq_path, _seq_name, "velodyne"
            )
            self.seqs_info[_seq_name]["img0_path"] = os.path.join(
                seq_path, _seq_name, "image_0"
            )
            self.seqs_info[_seq_name]["img1_path"] = os.path.join(
                seq_path, _seq_name, "image_1"
            )
            self.seqs_info[_seq_name]["calib_path"] = os.path.join(
                seq_path, _seq_name, calib_filename
            )
            self.seqs_info[_seq_name]["diff_gt_poses_path"] = os.path.join(
                seq_path, _seq_name, diff_gt_poses_filename
            )
            self.seqs_info[_seq_name]["diff_kiss_poses_path"] = os.path.join(
                seq_path, _seq_name, diff_kiss_poses_filename
            )
            if not (
                len(os.listdir(self.seqs_info[_seq_name]["img0_path"]))
                == len(os.listdir(self.seqs_info[_seq_name]["img1_path"]))
                == len(os.listdir(self.seqs_info[_seq_name]["pcd_path"]))
            ):
                raise RuntimeError(
                    "The number of point clouds and stereo images is not equal!"
                )

            self.seqs_info[_seq_name]["len"] = (
                len(os.listdir(self.seqs_info[_seq_name]["pcd_path"])) - 1
            )  # (len - 1) transforms per sequence

    def __len__(self):
        return sum([_seq_info["len"] for _seq_info in self.seqs_info])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
