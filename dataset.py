import os
import numpy as np
from torch.utils.data import Dataset
from typing import List
import skimage.io as io
from odom_utils import get_calibration

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class KittiVLOdomDataset(Dataset):
    """KITTI visual lidar odometry dataset."""

    def __init__(
        self,
        root_dir: str,
        calib_filename: str = "calib.txt",
        diff_gt_poses_filename: str = "diff_gt_poses.npy",
        diff_kiss_poses_filename: str = "diff_kiss_poses.npy",
        sequences_to_load: List[str] = [str(j).zfill(2) for j in range(11)],
        annotated: bool = True,
    ):
        """
        Arguments:
            root_dir (string): Root KITTI directory (dataset).
        """
        self.root_dir = root_dir

        # Initialize sequences information
        seq_path = os.path.join(root_dir, "sequences")
        seq_names = os.listdir(seq_path)

        if sequences_to_load is not None:
            if not set(sequences_to_load).issubset(seq_names):
                raise RuntimeError(
                    f"Not all sequences to load exist: {set(sequences_to_load).difference(seq_names)}!"
                )
            seq_names = sequences_to_load

        self.seqs_info = {
            seq: {
                prop: None
                for prop in [
                    "len",
                    "pcd_path",
                    "img0_path",
                    "img1_path",
                    "calib",
                    "diff_gt_poses",
                    "diff_kiss_poses",
                ]
            }
            for seq in seq_names
        }

        # Initialize sequences information
        for _seq_name in sorted(seq_names):
            self.seqs_info[_seq_name]["pcd_path"] = os.path.join(
                seq_path, _seq_name, "velodyne"
            )
            self.seqs_info[_seq_name]["img0_path"] = os.path.join(
                seq_path, _seq_name, "image_0"
            )
            self.seqs_info[_seq_name]["img1_path"] = os.path.join(
                seq_path, _seq_name, "image_1"
            )
            if not (
                len(os.listdir(self.seqs_info[_seq_name]["img0_path"]))
                == len(os.listdir(self.seqs_info[_seq_name]["img1_path"]))
                == len(os.listdir(self.seqs_info[_seq_name]["pcd_path"]))
            ):
                raise RuntimeError(
                    "The number of point clouds and stereo images is not equal!"
                )

            calib_path = os.path.join(seq_path, _seq_name, calib_filename)
            if not os.path.isfile(calib_path):
                raise RuntimeError(f"Calibration file does not exist: {calib_path}!")
            self.seqs_info[_seq_name]["calib"] = get_calibration(calib_path)

            diff_kiss_poses_path = os.path.join(
                seq_path, _seq_name, diff_kiss_poses_filename
            )
            if not os.path.isfile(diff_kiss_poses_path):
                raise RuntimeError(
                    f"Differential kiss poses do not exist: {diff_kiss_poses_path}!"
                )
            self.seqs_info[_seq_name]["diff_kiss_poses"] = np.load(diff_kiss_poses_path)

            diff_gt_poses_path = os.path.join(
                seq_path, _seq_name, diff_gt_poses_filename
            )
            if annotated and (not os.path.isfile(diff_gt_poses_path)):
                raise RuntimeError(
                    f"Differential gt poses do not exist: {diff_gt_poses_path}!"
                )
            self.seqs_info[_seq_name]["diff_gt_poses"] = (
                np.load(diff_gt_poses_path) if annotated else None
            )

            self.seqs_info[_seq_name]["len"] = (
                len(os.listdir(self.seqs_info[_seq_name]["pcd_path"])) - 1
            )  # (len - 1) transforms per sequence

    def __len__(self):
        return sum([_seq_info["len"] for _seq_info in self.seqs_info.values()])

    def __getitem__(self, idx):
        if self.__len__() <= idx or idx < 0:
            raise IndexError("Index is out of range!")

        cumsum = 0
        for _seq_info in self.seqs_info.values():
            if idx >= cumsum and idx < cumsum + _seq_info["len"]:
                break
            cumsum += _seq_info["len"]
        seq_idx = idx - cumsum

        img0_prev = io.imread(
            os.path.join(_seq_info["img0_path"], str(seq_idx).zfill(6)) + ".png"
        )
        img1_prev = io.imread(
            os.path.join(_seq_info["img1_path"], str(seq_idx).zfill(6)) + ".png"
        )
        pcd_prev = np.fromfile(
            os.path.join(_seq_info["pcd_path"], str(seq_idx).zfill(6)) + ".bin",
            dtype=np.float32,
        ).reshape(-1, 4)
        img0_curr = io.imread(
            os.path.join(_seq_info["img0_path"], str(seq_idx + 1).zfill(6)) + ".png"
        )
        img1_curr = io.imread(
            os.path.join(_seq_info["img1_path"], str(seq_idx + 1).zfill(6)) + ".png"
        )
        pcd_curr = np.fromfile(
            os.path.join(_seq_info["pcd_path"], str(seq_idx).zfill(6)) + ".bin",
            dtype=np.float32,
        ).reshape(-1, 4)

        return None


if __name__ == "__main__":
    dataset = KittiVLOdomDataset("data/SemanticKITTI/dataset")
    print(dataset[4539])
