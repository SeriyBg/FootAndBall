import torch
from torch import nn

class BallClassifierOpticalFlow(nn.Module):
    def __init__(self, lateral_channels, i_channels):
        super(BallClassifierOpticalFlow, self).__init__()

        # Feature extractor
        self.step1 = nn.Sequential(
            nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Flow processing
        self.flow_conv = nn.Sequential(
            nn.Conv2d(i_channels * 2, i_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final classification
        self.step2 = nn.Sequential(
            nn.Conv2d(2 * i_channels, i_channels, kernel_size=3, padding=1),  # Combine features and flow
            nn.ReLU(inplace=True),
            nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)  # Confidence map
        )

    def forward(self, x):
        """
        x: Current frame feature map.
        prev_x: Previous frame feature map (None for the first frame).
        """
        # Extract features
        x = self.step1(x)

        prev_x = torch.roll(x, shifts=1, dims=0).clone()
        prev_x[0] = torch.zeros_like(x[0].clone())
        next_x = torch.roll(x, shifts=-1, dims=0).clone()
        next_x[-1] = torch.zeros_like(x[-1].clone())

        flow_prev = x - prev_x
        flow_next = next_x - x

        # Concatenate both flows
        flow_features = torch.cat([flow_prev, flow_next], dim=1)
        flow = self.flow_conv(flow_features)

        # Concatenate features and flow
        x_fused = torch.cat([x, flow], dim=1)

        # Classify ball location
        output = self.step2(x_fused)

        return output
