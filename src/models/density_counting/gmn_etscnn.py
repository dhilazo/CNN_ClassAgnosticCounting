from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

from models.adapter import Adapter
from models.matching_module import MatchingModule
from models.variational_autoencoders.conv_vae import ConvVAE


class GmnETSCNN(nn.Module):
    """
    The GmnETSCNN follows the same architecture as the GMNETCNet but reduces the amount of learnable parameters
    thanks to using a siamese architecture.
    """

    def __init__(self, output_matching_size: Tuple[int, int] = None):
        super(GmnETSCNN, self).__init__()

        self.vae = ConvVAE()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.train(False)

        resnet_model = resnet50(pretrained=True)
        self.cut_resnet = nn.Sequential(*(list(resnet_model.children())[:6]))
        for module in self.cut_resnet.modules():
            if isinstance(module, Bottleneck) and isinstance(module.conv2, nn.Conv2d):
                module.conv2 = Adapter(module.conv2)

        self.adapt_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm2d(512)

        self.matching_model = MatchingModule(1024, output_matching_size)
        self.output_conv = nn.Conv2d(256, 1, 3, stride=1, padding=1)

    def forward(self, x: Tensor, x_object: Tensor, template: Tensor) -> Tensor:
        mu, logvar = self.vae.encoder(template)
        template_weights = self.vae.reparametrize(mu, logvar)
        template_weights = template_weights[0].repeat(3, 1, 1, 1)
        template_weights = template_weights.permute(1, 0, 2, 3)
        for module in self.cut_resnet.modules():
            if isinstance(module, nn.Conv2d):
                module.weight = torch.nn.Parameter(template_weights, requires_grad=False)
                break

        x_object = self.cut_resnet(x_object)
        x_object = self.adapt_pool(x_object)
        x_object = F.normalize(x_object, p=2, dim=-1)

        x = self.batch_norm(self.cut_resnet(x))

        x_object = x_object.repeat(1, 1, x.shape[-2], x.shape[-1])
        outputs = torch.cat((x, x_object), 1)

        outputs = self.matching_model(outputs)
        outputs = self.output_conv(outputs)
        return outputs

    def load_vae(self, path: str):
        self.vae.load_state_dict(torch.load(path))
