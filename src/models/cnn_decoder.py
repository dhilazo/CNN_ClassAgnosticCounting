import torch.nn as nn


class MyConvTranspose2d(nn.Module):
    """
    Custom Convolutional Transpose 2D layer which allows specifying the output size.

    Args:
        - conv (nn.ConvTranspose2d): Instance of a ConvTranspose2d layer.
        - output_size (Tuple, optional): Tuple indicating the desired output size of the layer.

    Attributes:
        - conv (nn.ConvTranspose2d): Where the conv arg is stored.
        - output_size (Tuple): Where the output_size arg is stored.
    """

    def __init__(self, conv: nn.ConvTranspose2d, output_size=None):
        super(MyConvTranspose2d, self).__init__()
        self.conv = conv
        self.output_size = output_size

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class CNNDecoder(nn.Module):
    """
    Convolutional Decoder module.

    Args:
        - bottleneck_size (int): input size of the first linear layer.
        - channels (int, optional): number of channels of the decoded image.
    """

    def __init__(self, channels=3):
        super(CNNDecoder, self).__init__()

        self.decoder = nn.Sequential(
            MyConvTranspose2d(nn.ConvTranspose2d(64, 32, 4), output_size=(10, 10)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # state size. 32 x 10 x 10
            MyConvTranspose2d(nn.ConvTranspose2d(32, 16, 5, stride=2, padding=1), output_size=(22, 22)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # state size. 16 x 22 x 22
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1), output_size=(46, 46)),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # state size. 128 x 26 x 21
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(8, channels, 7, stride=2, padding=1), output_size=(96, 96)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
