import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

path = "data/SemanticKITTI/dataset/sequences/00/diff_gt_poses_path.npy"

arr = np.load(path)
poses = [np.array([0, 0, 0])]
orients = [np.identity(3)]


for pq in arr:
    r_diff = R.from_quat(pq[3:]).as_matrix()
    t_diff = pq[:3]

    orients.append(r_diff @ orients[-1])
    poses.append((r_diff @ poses[-1]) + t_diff)


poses = np.array(poses)

real_poses = np.loadtxt("data/SemanticKITTI/dataset/poses/00.txt").reshape((-1, 3, 4))[
    :, :3, 3
]

print(real_poses.shape, poses.shape)
plt.plot(poses[:, 0], poses[:, 2])
plt.plot(real_poses[:, 0], real_poses[:, 2])
plt.show()


assert np.allclose(poses, real_poses, atol=1e-4)
