import torch
from torch import nn


class BallClassifier3D(nn.Module):
    def __init__(self, lateral_channels, i_channels):
        super(BallClassifier3D, self).__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.step2 = nn.Sequential(
            nn.Conv3d(i_channels, i_channels, kernel_size=(2, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        self.classifier = nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.step1(x)
        prev_x = torch.roll(x, shifts=1, dims=0).clone()
        prev_x[0] = x[0].clone()
        next_x = torch.roll(x, shifts=-1, dims=0).clone()
        next_x[-1] = x[-1].clone()
        assert torch.all(prev_x[1:] == x[:-1]), "prev_x is not correctly shifted"
        assert torch.all(prev_x[0] == x[0]), "prev_x[0] should be the same as x[0]"
        assert torch.all(next_x[:-1] == x[1:]), "next_x is not correctly shifted"
        assert torch.all(next_x[-1] == x[-1]), "next_x[-1] should be the same as x[-1]"
        x = torch.stack([prev_x, x, next_x], dim=2)
        x = self.step2(x)
        x = x.squeeze(2)
        x = self.classifier(x)
        return x
