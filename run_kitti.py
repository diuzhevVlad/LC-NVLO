import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from dataset import KittiVLOdomDataset
from model import SimpleStereoPoseNet
from torch.utils.data import DataLoader
import torch
import tqdm

path = "data/SemanticKITTI/dataset"
dataset = KittiVLOdomDataset(path, sequences_to_load=["05"])
loader = DataLoader(
    dataset,
    batch_size=14,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleStereoPoseNet(pretrained=False)
checkpoint = torch.load("checkpoints/checkpoint_epoch_9.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

real_poses = [np.array([0, 0, 0])]
real_orients = [np.identity(3)]
pred_poses = [np.array([0, 0, 0])]
pred_orients = [np.identity(3)]

with torch.no_grad():
    pbar = tqdm.tqdm(loader, leave=False)
    for batch_idx, batch in enumerate(pbar):
        left_t = batch["img0_prev"].to(device)
        right_t = batch["img1_prev"].to(device)
        left_t1 = batch["img0_curr"].to(device)
        right_t1 = batch["img1_curr"].to(device)
        kiss_prior = batch["kiss_prior"].to(device)

        gt = batch["gt"].numpy()  # (B,7)
        pred = model(left_t, right_t, left_t1, right_t1, kiss_prior).cpu().numpy()

        for i in range(gt.shape[0]):
            r_diff_real = R.from_quat(gt[i, 3:]).as_matrix()
            t_diff_real = gt[i, :3]
            real_orients.append(r_diff_real @ real_orients[-1])
            real_poses.append((r_diff_real @ real_poses[-1]) + t_diff_real)

            r_diff_pred = R.from_quat(pred[i, 3:]).as_matrix()
            t_diff_pred = pred[i, :3]
            pred_orients.append(r_diff_pred @ pred_orients[-1])
            pred_poses.append((r_diff_pred @ pred_poses[-1]) + t_diff_pred)

real_poses = np.array(real_poses)
pred_poses = np.array(pred_poses)

plt.plot(real_poses[:, 0], real_poses[:, 2])
plt.plot(pred_poses[:, 0], pred_poses[:, 2])
plt.show()


# assert np.allclose(poses, real_poses, atol=1e-4)
