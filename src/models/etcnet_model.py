import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from models.cnn_vae import ConvVAE


class FullyConnectedLayers(nn.Module):
    def __init__(self, output_size=128):
        super(FullyConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.training:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.training:
            x = self.dropout(x)

        return x


class ETCNet(nn.Module):
    """
    Encoded Template Convolution Network (ETCNet) is a neural network architecture based on the usage of a
    pretrained VAE which encodes a given template and uses its encoding as weights for the first convolution of a
    ResNet.
    """

    def __init__(self, output_size=10):
        super(ETCNet, self).__init__()

        self.vae = ConvVAE()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.train(False)

        self.resnet_model = resnet50(pretrained=True)
        self.resnet_model.fc = FullyConnectedLayers()
        self.output = nn.Linear(128, output_size)

    def forward(self, x, x_object):
        mu, logvar = self.vae.encoder(x_object)
        template_weights = self.vae.reparametrize(mu, logvar)  # TODO create 3 VAE 1 per channel
        template_weights = template_weights[0].repeat(3, 1, 1, 1)
        template_weights = template_weights.permute(1, 0, 2, 3)
        self.resnet_model.conv1.weight = torch.nn.Parameter(template_weights, requires_grad=False)
        x = self.resnet_model(x)
        x = self.output(x)
        return x

    def load_vae(self, path):
        self.vae.load_state_dict(torch.load(path))
