from torch import nn

class BallClassifierWithAttention(nn.Module):
    def __init__(self, lateral_channels, i_channels, dropout_rate=0.3):
        super(BallClassifierWithAttention, self).__init__()

        # Feature extraction
        self.step1 = nn.Sequential(
            nn.Conv2d(lateral_channels, i_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(i_channels),  # Normalization
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate)
        )

        # Attention Layer (Spatial Attention)
        self.attention = nn.Sequential(
            nn.Conv2d(i_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.step2 = nn.Sequential(
            nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)
        )
        # Ball Classification
        # self.step2 = nn.Sequential(
        #     nn.Conv2d(i_channels, i_channels // 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(i_channels // 2),  # Normalization
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_rate),  # Dropout
        #     nn.Conv2d(i_channels // 2, out_channels=2, kernel_size=3, padding=1)
        # )

    def forward(self, x):
        x = self.step1(x)  # Extract features

        attn_map = self.attention(x)  # Compute attention map
        x = x * attn_map  # Apply attention

        x = self.step2(x)  # Final classification
        return x
