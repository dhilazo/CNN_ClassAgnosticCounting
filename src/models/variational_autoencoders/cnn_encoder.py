from typing import Tuple

import torch.nn as nn
from torch import Tensor


class CNNEncoder(nn.Module):
    """
    Convolutional Encoder module.

    Args:
        - bottleneck_size (int): output size of the last linear layer.
        - channels (int, optional): number of channels of the decoded image.
    """

    def __init__(self, channels: int = 3):
        super(CNNEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # input is channels x 96 x 96
            nn.Conv2d(channels, 8, 7, stride=2, padding=1),
            nn.ReLU(),
            # state size. 8 x 46 x 46
            nn.Conv2d(8, 16, 5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 22 x 22
            nn.Conv2d(16, 32, 5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # state size. 32 x 10 x 10
        )
        self.last_conv1 = nn.Sequential(nn.Conv2d(32, 64, 4), nn.BatchNorm2d(64), nn.ReLU())
        self.last_conv2 = nn.Sequential(nn.Conv2d(32, 64, 4), nn.BatchNorm2d(64), nn.ReLU())
        # state size. 64 x 7 x 7

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        return self.last_conv1(x), self.last_conv1(x)
