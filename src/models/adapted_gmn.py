import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
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
        original = x
        adapt = self.adapter(x)
        x = self.conv(x)

        x += adapt
        return x


class AdaptedGenericMatchingNetwork(nn.Module):
    """

    """

    def __init__(self, output_matching_size=None):
        super(AdaptedGenericMatchingNetwork, self).__init__()

        resnet_model = resnet50(pretrained=True)
        self.cut_resnet = nn.Sequential(*(list(resnet_model.children())[:6]))
        for module in self.cut_resnet.modules():
            if isinstance(module, Bottleneck) and isinstance(module.conv2, nn.Conv2d):
                module.conv2 = Adapter(module.conv2)

        resnet_model = resnet50(pretrained=True)
        self.cut_resnet_template = nn.Sequential(*(list(resnet_model.children())[:6]))
        for module in self.cut_resnet_template.modules():
            if isinstance(module, Bottleneck) and isinstance(module.conv2, nn.Conv2d):
                module.conv2 = Adapter(module.conv2)

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm2d(512)

        self.matching_model = MatchingNet(1024, output_matching_size)
        self.output_conv = nn.Conv2d(256, 1, 3, stride=1, padding=1)  # TODO compute padding same

    def forward(self, x, x_object):
        x_object = self.cut_resnet_template(x_object)
        x_object = self.adapt_pool(x_object)
        x_object = F.normalize(x_object, p=2, dim=-1)

        x = self.batch_norm(self.cut_resnet(x))

        x_object = x_object.repeat(1, 1, x.shape[-2], x.shape[-1])
        outputs = torch.cat((x, x_object), 1)

        outputs = self.matching_model(outputs)
        outputs = self.output_conv(outputs)
        return outputs

    @staticmethod
    def get_count(matrix, plot=False):
        footprint_3x3 = np.ones((3, 3))
        footprint_3x3[0, 0] = 0
        footprint_3x3[2, 0] = 0
        footprint_3x3[0, 2] = 0
        footprint_3x3[2, 2] = 0

        footprint_7x7 = np.ones((7, 7))
        footprint_7x7[0, 0] = 0
        footprint_7x7[6, 0] = 0
        footprint_7x7[0, 6] = 0
        footprint_7x7[6, 6] = 0

        # Small maximum analysis
        data_max = ndimage.maximum_filter(matrix, footprint=footprint_3x3)
        maxima = (matrix == data_max)
        data_min = ndimage.minimum_filter(matrix, footprint=footprint_3x3)
        diff = ((data_max - data_min) > (matrix.max() / 3))
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) // 2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1) // 2
            y.append(y_center)

        coordinates_small = list(zip(x, y))

        # Big maximum analysis
        data_max = ndimage.maximum_filter(matrix, footprint=footprint_7x7)
        maxima = (matrix == data_max)
        data_min = ndimage.minimum_filter(matrix, footprint=footprint_7x7)
        diff = ((data_max - data_min) > (matrix.max() / 3))
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) // 2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1) // 2
            y.append(y_center)

        coordinates_big = list(zip(x, y))

        coordinates = np.array(
            list(set(coordinates_big + coordinates_small)))

        if plot:
            plt.figure()
            plt.imshow(matrix, cmap="gray")
            plt.axis('off')
            plt.autoscale(False)
            plt.plot(coordinates[:, 0], coordinates[:, 1], 'rx')
            # plt.title("Located instances")
            plt.show()

        return len(coordinates)
