import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

from models.adapter import Adapter
from models.matching_module import MatchingModule


class AdaptedGenericMatchingNetwork(nn.Module):
    """
    The AdaptedGenericMatchingNetwork is a Generic Matching Network (GMN) with adapter convolutional
    connections at the end of each block of layer in its ResNet.
    """

    def __init__(self, output_matching_size: Tuple[int, int] = None):
        super(AdaptedGenericMatchingNetwork, self).__init__()

        resnet_model = resnet50(pretrained=True)
        self.cut_resnet = nn.Sequential(*(list(resnet_model.children())[:6]))
        for module in self.cut_resnet.modules():
            if isinstance(module, Bottleneck) and isinstance(module.conv2, nn.Conv2d):
                module.conv2 = Adapter(module.conv2)

        self.cut_resnet_template = copy.deepcopy(self.cut_resnet)

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm2d(512)

        self.matching_model = MatchingModule(1024, output_matching_size)
        self.output_conv = nn.Conv2d(256, 1, 3, stride=1, padding=1)

    def forward(self, x: Tensor, x_object: Tensor, template: Tensor) -> Tensor:
        x_object = self.cut_resnet_template(x_object)
        x_object = self.adapt_pool(x_object)
        x_object = F.normalize(x_object, p=2, dim=-1)

        x = self.batch_norm(self.cut_resnet(x))

        x_object = x_object.repeat(1, 1, x.shape[-2], x.shape[-1])
        outputs = torch.cat((x, x_object), 1)

        outputs = self.matching_model(outputs)
        outputs = self.output_conv(outputs)
        return outputs
