import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------
# 2) A small MLP to fuse stereo or temporal features
# ---------------------------------------------------------
class SimpleMLP(nn.Module):
    """
    A generic 2-layer MLP for demonstration.
    Input dimension -> hidden -> output dimension
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------
# 3) Main Architecture
# ---------------------------------------------------------
class SimpleStereoPoseNet(nn.Module):
    """
    A simplified model that:
      1) Uses ResNet18 as a shared feature extractor for each image.
      2) Fuses left/right features at time t into a vector -> fuse_stereo_t
      3) Fuses left/right features at time t+1 into another vector -> fuse_stereo_t1
      4) Fuses the two time vectors -> fuse_temp
      5) Concatenates that fused vector with the LiDAR prior (7D)
      6) Outputs a 7D correction (3D trans + 4D quaternion), normalizes the quaternion
    """

    def __init__(self, pretrained=False):
        super().__init__()

        # ---- Shared ResNet18 for 1-channel inputs ----
        base_resnet = models.resnet18(
            weights=(
                models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else models.ResNet18_Weights.DEFAULT
            )
        )

        # We'll discard the fully connected (fc) layer, keep only the "backbone"
        # We will manually do global average pooling and an MLP if needed.
        layers = list(base_resnet.children())[:-1]
        # This leaves us with [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # Actually, the last element is "avgpool" in PyTorch's ResNet, so let's remove that too if we want to do it manually:
        if isinstance(layers[-1], nn.AdaptiveAvgPool2d):
            layers = layers[:-1]
        self.backbone = nn.Sequential(*layers)  # The output shape is (B, 512, H', W')

        # We do global average pooling ourselves in forward.

        # ---- MLP to fuse stereo features at each timestep ----
        # We'll take 512-d from backbone's global pool => 512 dims
        # For both left & right, we have two 512-d vectors => 1024
        # We'll reduce it to 256 dims for each time step
        self.mlp_stereo = SimpleMLP(in_dim=1024, hidden_dim=512, out_dim=256)

        # ---- MLP to fuse the two time-step vectors (256 + 256 = 512) -> 256
        self.mlp_time = SimpleMLP(in_dim=512, hidden_dim=512, out_dim=256)

        # ---- Final pose head: (256 + 7 [LiDAR prior] = 263) -> 7
        self.mlp_pose = SimpleMLP(in_dim=263, hidden_dim=256, out_dim=7)

    def encode_image(self, x):
        """
        Pass single grayscale image through ResNet18 backbone
        and do global average pooling => returns a 512-d vector.
        x shape: (B,1,H,W)
        """
        feat_map = self.backbone(x)  # (B,512,H',W') typically H'=W'=something smaller
        # Global average pool
        # For standard ResNet18 with input ~ (B,1,224,224), final is (B,512,7,7)
        pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))  # (B,512,1,1)
        vec = pooled.view(pooled.size(0), -1)  # (B,512)
        return vec

    def fuse_stereo(self, left_vec, right_vec):
        """
        Fuse stereo features by simple concatenation -> MLP -> out 256-d
        left_vec, right_vec: (B,512) each
        output: (B,256)
        """
        stereo_input = torch.cat([left_vec, right_vec], dim=1)  # (B,1024)
        stereo_out = self.mlp_stereo(stereo_input)  # (B,256)
        return stereo_out

    def fuse_time(self, emb_t, emb_t1):
        """
        Fuse the embeddings from time t and t+1
        emb_t, emb_t1: (B,256) each
        output: (B,256)
        """
        time_input = torch.cat([emb_t, emb_t1], dim=1)  # (B,512)
        time_out = self.mlp_time(time_input)  # (B,256)
        return time_out

    def forward(self, left_t, right_t, left_t1, right_t1, lidar_prior):
        """
        Inputs:
          left_t, right_t:   (B,1,H,W) stereo images at time t
          left_t1, right_t1: (B,1,H,W) stereo images at time t+1
          lidar_prior:       (B,7)  => (tx,ty,tz, qw,qx,qy,qz)

        Output:
          delta_pose: (B,7) => (dx,dy,dz, qw,qx,qy,qz) [quat normalized]
        """
        # 1) Encode each image -> 512-d
        left_t_vec = self.encode_image(left_t)  # (B,512)
        right_t_vec = self.encode_image(right_t)  # (B,512)
        left_t1_vec = self.encode_image(left_t1)  # (B,512)
        right_t1_vec = self.encode_image(right_t1)  # (B,512)

        # 2) Fuse stereo at time t -> 256-d
        emb_t = self.fuse_stereo(left_t_vec, right_t_vec)  # (B,256)

        # 3) Fuse stereo at time t+1 -> 256-d
        emb_t1 = self.fuse_stereo(left_t1_vec, right_t1_vec)  # (B,256)

        # 4) Fuse across time -> 256-d
        fused_temp = self.fuse_time(emb_t, emb_t1)  # (B,256)

        # 5) Combine with LiDAR prior -> feed to final MLP -> 7-d
        combined = torch.cat([fused_temp, lidar_prior], dim=1)  # (B,256 + 7) = (B,263)
        delta_pose_raw = self.mlp_pose(combined)  # (B,7)

        # 6) Normalize quaternion
        trans = delta_pose_raw[:, :3]  # (B,3)
        quat_raw = delta_pose_raw[:, 3:]  # (B,4)
        quat_norm = quat_raw / quat_raw.norm(
            dim=1, keepdim=True
        )  # ensure unit quaternion
        delta_pose = torch.cat([trans, quat_norm], dim=1)  # (B,7)

        return delta_pose


# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create a small dummy batch
    B, H, W = 2, 376, 1241
    left_t = torch.randn(B, 3, H, W)
    right_t = torch.randn(B, 3, H, W)
    left_t1 = torch.randn(B, 3, H, W)
    right_t1 = torch.randn(B, 3, H, W)

    # 7D LiDAR prior: (tx, ty, tz, qw, qx, qy, qz)
    lidar_prior = torch.randn(B, 7)

    # Instantiate the model
    model = SimpleStereoPoseNet(pretrained=True)

    # Forward pass
    out = model(left_t, right_t, left_t1, right_t1, lidar_prior)
    print("Output shape:", out.shape)  # should be (B, 7)
    print("Sample output:", out)
