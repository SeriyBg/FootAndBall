import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms.functional as TF


class BallClassifierOpticalFlow(nn.Module):
    def __init__(self, lateral_channels, i_channels):
        super(BallClassifierOpticalFlow, self).__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * i_channels, i_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x, flow):
        """
        x: (B, C, H, W) - Current frame features
        flow: (B, 2, H, W) - Optical flow from prev to current frame
        """
        x = self.step1(x)  # Extract initial features from current frame

        # Shift previous frame features using optical flow
        prev_x = torch.roll(x, shifts=1, dims=0).clone()
        prev_x[0] = x[0].clone()  # Use the same frame for the first input

        # Warp the previous frame using optical flow
        prev_x = self.warp_features(prev_x, flow)

        # Concatenate current and warped previous frame features
        x = torch.cat([x, prev_x], dim=1)  # (B, 2 * i_channels, H, W)

        # Refinement through Conv2D
        x = self.fusion_conv(x)
        x = self.final_conv(x)

        return x

    def warp_features(self, features, flow):
        """
        Warps the features based on the optical flow.
        :param features: (B, C, H, W) - Feature map from previous frame
        :param flow: (B, 2, H, W) - Optical flow vectors
        :return: Warped features (B, C, H, W)
        """
        B, C, H, W = features.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(features.device)  # (2, H, W)

        # Normalize flow for grid_sample
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        warped_grid = grid + flow  # Apply flow displacement
        warped_grid = warped_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Normalize to [-1, 1] for grid_sample
        warped_grid[..., 0] = (warped_grid[..., 0] / (W - 1)) * 2 - 1
        warped_grid[..., 1] = (warped_grid[..., 1] / (H - 1)) * 2 - 1

        # Warp features using bilinear interpolation
        warped_features = F.grid_sample(features, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped_features
