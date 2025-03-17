import torch
from torch import nn


class BallClassifierFusion(nn.Module):
    def __init__(self, lateral_channels, i_channels):
        super(BallClassifierFusion, self).__init__()
        self.conv = nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_fusion = nn.Sequential(nn.Conv2d(i_channels * 3, out_channels=i_channels, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1))


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        prev_x = torch.roll(x, shifts=1, dims=0).clone()
        prev_x[0] = x[0].clone()
        next_x = torch.roll(x, shifts=-1, dims=0).clone()
        next_x[-1] = x[-1].clone()
        assert torch.all(prev_x[1:] == x[:-1]), "prev_x is not correctly shifted"
        assert torch.all(prev_x[0] == x[0]), "prev_x[0] should be the same as x[0]"
        assert torch.all(next_x[:-1] == x[1:]), "next_x is not correctly shifted"
        assert torch.all(next_x[-1] == x[-1]), "next_x[-1] should be the same as x[-1]"
        x = torch.cat([x, prev_x, next_x], dim=1)
        x = self.conv_fusion(x)
        return x
