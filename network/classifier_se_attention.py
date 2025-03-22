import torch
from torch import nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ClassifierWithSEAttention(nn.Module):
    def __init__(self, lateral_channels, i_channels, dropout_rate=0.3, reduction=16):
        super(ClassifierWithSEAttention, self).__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(i_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate)
        )
        self.se_block = SEBlock(i_channels, reduction=reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(i_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.step2 = nn.Sequential(
            nn.Conv2d(i_channels, i_channels // 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(i_channels // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Conv2d(i_channels // 2, out_channels=2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.step1(x)
        x = self.se_block(x)
        attn_map = self.spatial_attention(x)
        x = x * attn_map
        x = self.step2(x)
        return x
