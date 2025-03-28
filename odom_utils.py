import numpy as np
from scipy.spatial.transform import Rotation as R


def check_spatial_transform(T: np.ndarray) -> None:
    if not (T.shape == (4, 4)):
        raise ValueError("Non-4x4 matrix is not a spatial transformation!")
    if np.any(np.real(T) != T):
        raise ValueError("Only real numbers are supported in spatial transformations!")
    if not np.allclose(T[:3, :3] @ T.T[:3, :3], np.identity(3), atol=1e-4):
        raise ValueError("Left-upper 3x3 matrix is not rotation!")
    if np.any(T[3, :] != np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("The footer must be [0 0 0 1]!")


def get_diff_transform(T_prev: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
    check_spatial_transform(T_prev)
    check_spatial_transform(T_curr)
    return T_curr @ np.linalg.inv(T_prev)


def transform_to_pq(T: np.ndarray) -> np.ndarray:
    check_spatial_transform(T)
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()])


def get_calibration(calib_path: str):
    with open(calib_path) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(" ")[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(" ")[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(" ")[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    return {
        "P0": P0.reshape(3, 4),
        "P1": P1.reshape(3, 4),
        "K0": P0.reshape(3, 4)[:3, :3].copy(),
        "K1": P1.reshape(3, 4)[:3, :3].copy(),
        "Tr": Tr_velo_to_cam.reshape(3, 4),
    }
