import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

cfg = {
    'X': [16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
}

se_layers = [0]

def make_modules(cfg, batch_norm=False):
    modules = nn.ModuleList()
    out_channels = []

    in_channels = 3
    layers = []

    for idx, v in enumerate(cfg):
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            modules.append(nn.Sequential(*layers))
            out_channels.append(in_channels)
            layers = []
        else:
            if idx in se_layers:
                # Apply SE to the first convolution layer
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, SEBlock(v), nn.ReLU(inplace=True)]
            else:
                if batch_norm:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    assert len(layers) == 0  # Ensure no leftover layers

    return modules, out_channels

class FPN(nn.Module):
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None):
        super(FPN, self).__init__()
        assert len(layers) == len(out_channels)

        self.layers = layers
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.lateral_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        self.return_layers = return_layers if return_layers is not None else list(range(len(layers)-1))
        self.min_returned_layer = min(self.return_layers)

        # Create lateral layers for channel reduction
        for i in range(self.min_returned_layer, len(self.layers)):
            self.lateral_layers.append(nn.Conv2d(out_channels[i], self.lateral_channels, kernel_size=1, stride=1, padding=0))

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up pass: store feature maps
        c = []
        for m in self.layers:
            x = m(x)
            c.append(x)

        # Top-down pass with lateral layers
        p = [self.lateral_layers[-1](c[-1])]
        for i in range(len(c)-2, self.min_returned_layer-1, -1):
            temp = self._upsample_add(p[-1], self.lateral_layers[i-self.min_returned_layer](c[i]))
            p.append(temp)

        p = p[::-1]  # Reverse order

        out_tensors = [p[l - self.min_returned_layer] for l in self.return_layers]
        return out_tensors