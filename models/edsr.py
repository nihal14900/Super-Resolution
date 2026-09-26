import torch
from torch import nn

from utils.count_parameters import print_parameters

class ResidualBlock(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_features, num_features, 3, 1, 1)
        )

    def forward(self, x):

        x = x + self.body(x)

        return x

class UpSampler(nn.Sequential):

    def __init__(self, scale, num_features):

        layers = []

        if scale in (2, 4):
            for _ in range(scale // 2):
                layers.append(nn.Conv2d(num_features, 4 * num_features, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))

        elif scale == 3:
            layers.append(nn.Conv2d(num_features, 9 * num_features, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))

        super().__init__(*layers)

class EDSR(nn.Module):

    def __init__(self, scale, num_blocks, num_features):
        super().__init__()

        self.head = nn.Conv2d(3, num_features, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)],
            nn.Conv2d(num_features, num_features, 3, 1, 1)
        )
        self.upsampler = UpSampler(scale, num_features)
        self.tail = nn.Conv2d(num_features, 3, 3, 1, 1)

    def forward(self, x):
        shallow_features = self.head(x)
        deep_features = shallow_features + self.body(shallow_features)
        upsampled_features = self.upsampler(deep_features)

        return self.tail(upsampled_features)
