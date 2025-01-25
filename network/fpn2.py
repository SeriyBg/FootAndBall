# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import torch.nn as nn
import torch.nn.functional as F


cfg = {
    # Config according to the Table 1 from the FootAndBall paper
    'X': [16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
}


def make_modules(cfg, batch_norm=False):
    # Each module is a list of sequential layers operating at the same spatial dimension followed by MaxPool2d
    modules = nn.ModuleList()
    # Number of output channels in each module
    out_channels = []

    in_channels = 3
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # Create new module with accumulated layers and flush layers list
            modules.append(nn.Sequential(*layers))
            out_channels.append(in_channels)
            layers = []
        else:
            if batch_norm:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    # 'M' should be the last layer - and all layers should be flushed
    assert len(layers) == 0

    return modules, out_channels


class FPN(nn.Module):
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None):
        # return_layers: index of layers (numbered from 0) for which feature maps are returned
        super(FPN, self).__init__()

        assert len(layers) == len(out_channels)

        self.layers = layers
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.lateral_layers = nn.ModuleList()
        self.transpose_layers = nn.ModuleList()  # Transpose convolution layers
        if return_layers is None:
            self.return_layers = list(range(len(layers) - 1))
        else:
            self.return_layers = return_layers
        self.min_returned_layer = min(self.return_layers)

        for i in range(self.min_returned_layer, len(self.layers)):
            self.lateral_layers.append(nn.Conv2d(out_channels[i], self.lateral_channels, kernel_size=1, stride=1,
                                                 padding=0))
            self.transpose_layers.append(
                nn.ConvTranspose2d(
                    self.lateral_channels,
                    self.lateral_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    output_padding=1
                )
            )

    def _upsample_add(self, x, y, upsample_layer):
        '''Upsample using transpose convolution and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
          upsample_layer: (nn.ConvTranspose2d) the layer to perform upsampling.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        x_upsampled = upsample_layer(x, output_size=(H, W))  # Use output_size to explicitly match dimensions
        return x_upsampled + y

    def forward(self, x):
        # Bottom-up pass, store all intermediary feature maps in list c
        c = []
        for m in self.layers:
            x = m(x)
            c.append(x)

        # Top-down pass
        p = [self.lateral_layers[-1](c[-1])]

        for i in range(len(c) - 2, self.min_returned_layer - 1, -1):
            temp = self._upsample_add(
                p[-1],
                self.lateral_layers[i - self.min_returned_layer](c[i]),
                self.transpose_layers[i - self.min_returned_layer]  # Use corresponding transpose convolution layer
            )
            p.append(temp)

        # Reverse the order of tensors in p
        p = p[::-1]

        out_tensors = []
        for ndx, l in enumerate(self.return_layers):
            temp = p[l - self.min_returned_layer]
            out_tensors.append(temp)

        return out_tensors
