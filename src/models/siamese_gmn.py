import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


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


class MatchingNet(nn.Module):
    def __init__(self, channels, output_size):
        super(MatchingNet, self).__init__()
        # TODO compute padding same
        self.matching_model = nn.Sequential(nn.Conv2d(channels, 256, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            MyConvTranspose2d(nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1),
                                                              output_size=output_size),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU())

    def forward(self, x):
        return self.matching_model(x)


class Adapter(nn.Module):
    def __init__(self, conv: nn.Conv2d, ):
        super(Adapter, self).__init__()
        self.conv = conv
        self.adapter = nn.Conv2d(conv.in_channels, conv.out_channels, 1, stride=self.conv.stride)

    def forward(self, x):
        adapt = self.adapter(x)
        x = self.conv(x)

        x += adapt
        return x


class SiameseGenericMatchingNetwork(nn.Module):
    """

    """

    def __init__(self, output_matching_size=None):
        super(SiameseGenericMatchingNetwork, self).__init__()

        resnet_model = resnet50(pretrained=True)
        self.cut_resnet = nn.Sequential(*(list(resnet_model.children())[:6]))
        for module in self.cut_resnet.modules():
            if isinstance(module, Bottleneck) and isinstance(module.conv2, nn.Conv2d):
                module.conv2 = Adapter(module.conv2)

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm2d(512)

        self.matching_model = MatchingNet(1024, output_matching_size)
        self.output_conv = nn.Conv2d(256, 1, 3, stride=1, padding=1)  # TODO compute padding same

    def forward(self, x, x_object):
        x_object = self.cut_resnet(x_object)
        x_object = self.adapt_pool(x_object)
        x_object = F.normalize(x_object, p=2, dim=-1)

        x = self.batch_norm(self.cut_resnet(x))

        x_object = x_object.repeat(1, 1, x.shape[-2], x.shape[-1])
        outputs = torch.cat((x, x_object), 1)

        outputs = self.matching_model(outputs)
        outputs = self.output_conv(outputs)
        return outputs
