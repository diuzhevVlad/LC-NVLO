import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# For TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# TQDM for progress bars
from tqdm import tqdm

# Assuming dataset.py and model.py are in the same directory:
from dataset import KittiVLOdomDataset
from model import SimpleStereoPoseNet


def pose_loss(pred, gt):
    """
    A simple combined MSE loss for translation & quaternion.
    pred, gt: (B,7) => (tx, ty, tz, qw, qx, qy, qz)
    Returns a scalar loss.
    """
    # Split into translation vs rotation
    trans_pred = pred[:, :3]
    trans_gt = gt[:, :3]
    quat_pred = pred[:, 3:]
    quat_gt = gt[:, 3:]

    # L2 loss on translation
    loss_trans = nn.functional.mse_loss(trans_pred, trans_gt)
    # L2 loss on quaternion
    loss_quat = nn.functional.mse_loss(quat_pred, quat_gt)

    # Combine (you could weight these differently or use a geodesic loss for orientation)
    loss = loss_trans + loss_quat
    return loss


def train_one_epoch(
    model, dataloader, optimizer, device, epoch_idx=None, num_epochs=None
):
    """
    Train for one epoch. Returns average loss over the entire dataset.
    Uses TQDM to display progress.
    """
    model.train()
    total_loss = 0.0

    # Create a tqdm wrapper for the dataloader
    loop_desc = (
        f"Train Epoch [{epoch_idx}/{num_epochs}]"
        if (epoch_idx and num_epochs)
        else "Training"
    )
    pbar = tqdm(dataloader, desc=loop_desc, leave=False)

    for batch_idx, batch in enumerate(pbar):
        # batch keys: img0_prev, img1_prev, img0_curr, img1_curr, kiss_prior, gt
        left_t = batch["img0_prev"].to(device)
        right_t = batch["img1_prev"].to(device)
        left_t1 = batch["img0_curr"].to(device)
        right_t1 = batch["img1_curr"].to(device)
        kiss_prior = batch["kiss_prior"].to(device)
        gt = batch["gt"].to(device)  # (B,7)

        optimizer.zero_grad()
        # Forward
        pred = model(left_t, right_t, left_t1, right_t1, kiss_prior)
        # Compute loss
        loss = pose_loss(pred, gt)
        # Backprop
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss * left_t.size(0)  # accumulate total for average

        # Update tqdm bar
        pbar.set_postfix(
            {
                "Batch": batch_idx,
                "Loss": f"{current_loss:.4f}",
            }
        )

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, device, epoch_idx=None, num_epochs=None):
    """
    Evaluate on validation set. Returns mean translation error & mean quaternion error.
    Uses TQDM to display progress.
    """
    model.eval()
    total_translation_error = 0.0
    total_quat_error = 0.0
    count = 0

    loop_desc = (
        f"Val Epoch [{epoch_idx}/{num_epochs}]"
        if (epoch_idx and num_epochs)
        else "Validation"
    )
    pbar = tqdm(dataloader, desc=loop_desc, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            left_t = batch["img0_prev"].to(device)
            right_t = batch["img1_prev"].to(device)
            left_t1 = batch["img0_curr"].to(device)
            right_t1 = batch["img1_curr"].to(device)
            kiss_prior = batch["kiss_prior"].to(device)
            gt = batch["gt"].to(device)  # (B,7)

            pred = model(left_t, right_t, left_t1, right_t1, kiss_prior)

            # Compute translation error (Euclidean distance) per sample
            trans_error = (pred[:, :3] - gt[:, :3]).pow(2).sum(dim=1).sqrt()  # (B,)
            total_translation_error += trans_error.sum().item()

            # Quaternion L2 distance in 4D space
            quat_error = (pred[:, 3:] - gt[:, 3:]).pow(2).sum(dim=1).sqrt()  # (B,)
            total_quat_error += quat_error.sum().item()

            count += pred.size(0)

            # For TQDM, we can display partial progress if desired
            pbar.set_postfix(
                {
                    "Batch": batch_idx,
                    "SamplesProcessed": count,
                }
            )

    mean_translation_error = total_translation_error / count
    mean_quat_error = total_quat_error / count
    return mean_translation_error, mean_quat_error


def main():
    # ------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------
    root_dir = "data/SemanticKITTI/dataset"  # Adjust to your actual dataset root
    train_sequences = [f"{i:02d}" for i in range(7)]  # 00..08
    val_sequences = [f"{i:02d}" for i in range(7, 11)]  # 09..10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 14
    num_workers = 4
    lr = 1e-4
    start_epoch = 6
    num_epochs = 20  # Adjust as needed
    start_checkpoint = "checkpoints/checkpoint_epoch_5.pth"

    # TensorBoard log directory
    log_dir = "./logs/stereo_pose_experiment"
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------
    # 2. Create Datasets & DataLoaders
    # ------------------------------------------------
    train_dataset = KittiVLOdomDataset(
        root_dir=root_dir, sequences_to_load=train_sequences, annotated=True
    )
    val_dataset = KittiVLOdomDataset(
        root_dir=root_dir, sequences_to_load=val_sequences, annotated=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # ------------------------------------------------
    # 3. Create Model, Optimizer
    # ------------------------------------------------
    model = SimpleStereoPoseNet(pretrained=True)
    checkpoint = torch.load(start_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------
    # 4. Training/Evaluation Loop with TQDM + TensorBoard
    # ------------------------------------------------
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # ---- Train ----
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch_idx=epoch,
            num_epochs=start_epoch + num_epochs,
        )

        # ---- Evaluate ----
        val_trans_err, val_quat_err = evaluate(
            model, val_loader, device, epoch_idx=epoch, num_epochs=num_epochs
        )

        # ---- Print & Log to TensorBoard ----
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Trans Err: {val_trans_err:.4f} | "
            f"Val Quat Err: {val_quat_err:.4f}"
        )

        # Log to TensorBoard
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/TranslationError", val_trans_err, epoch)
        writer.add_scalar("Val/QuaternionError", val_quat_err, epoch)

        ckpt_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

    writer.close()
    print("Training complete. TensorBoard logs saved to:", log_dir)


if __name__ == "__main__":
    main()
