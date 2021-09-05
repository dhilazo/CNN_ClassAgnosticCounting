from typing import Tuple

import torch.nn as nn
from torch import Tensor

from models.custom_conv_transpose2d import CustomConvTranspose2d


class MatchingModule(nn.Module):
    def __init__(self, channels: int, output_size: Tuple[int, int]):
        super(MatchingModule, self).__init__()
        self.matching_model = nn.Sequential(
            nn.Conv2d(channels, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CustomConvTranspose2d(256, 256, 3, stride=2, padding=1, output_size=output_size),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.matching_model(x)
