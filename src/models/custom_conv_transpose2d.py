from typing import Tuple

import torch.nn as nn
from torch import Tensor


class CustomConvTranspose2d(nn.Module):
    """
    Custom Convolutional Transpose 2D layer which allows specifying the output size.

    Args:
        - output_size (Tuple, optional): Tuple indicating the desired output size of the layer.

    Attributes:
        - conv (nn.ConvTranspose2d): Instance of a ConvTranspose2d layer.
        - output_size (Tuple): Where the output_size arg is stored.
    """

    def __init__(self, *args, output_size: Tuple[int, int] = None, **kwargs):
        super(CustomConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x, output_size=self.output_size)
        return x
