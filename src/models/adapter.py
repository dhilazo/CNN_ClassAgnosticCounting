import torch.nn as nn
from torch import Tensor


class Adapter(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
    ):
        super(Adapter, self).__init__()
        self.conv = conv
        self.adapter = nn.Conv2d(conv.in_channels, conv.out_channels, 1, stride=self.conv.stride)

    def forward(self, x: Tensor) -> Tensor:
        adapt = self.adapter(x)
        x = self.conv(x)

        x += adapt
        return x
