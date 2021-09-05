import torch.nn as nn
from torch import Tensor

from models.custom_conv_transpose2d import CustomConvTranspose2d


class CNNDecoder(nn.Module):
    """
    Convolutional Decoder module.

    Args:
        - bottleneck_size (int): input size of the first linear layer.
        - channels (int, optional): number of channels of the decoded image.
    """

    def __init__(self, channels: int = 3):
        super(CNNDecoder, self).__init__()

        self.decoder = nn.Sequential(
            CustomConvTranspose2d(64, 32, 4, output_size=(10, 10)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # state size. 32 x 10 x 10
            CustomConvTranspose2d(32, 16, 5, stride=2, padding=1, output_size=(22, 22)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 22 x 22
            CustomConvTranspose2d(16, 8, 5, stride=2, padding=1, output_size=(46, 46)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # state size. 128 x 26 x 21
            CustomConvTranspose2d(8, channels, 7, stride=2, padding=1, output_size=(96, 96)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        return x
