import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50

from models.variational_autoencoders.conv_vae import ConvVAE


class FullyConnectedLayers(nn.Module):
    def __init__(self, output_size: int = 128):
        super(FullyConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor) -> Tensor:
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


class ETSCNN(nn.Module):
    """
    Encoded Template Siamese Convolutional Neural Network (ETSCNN) is a neural network architecture based on the usage
    of a pretrained VAE which encodes a given template and uses its encoding as weights for the first convolution of a
    ResNet. Then it uses the properties of Siamese NN to pass both inputs through the network.
    """

    def __init__(self, output_size: int = 10):
        super(ETSCNN, self).__init__()

        self.vae = ConvVAE()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.train(False)

        self.resnet_model = resnet50(pretrained=True)
        self.resnet_model.fc = FullyConnectedLayers()
        self.output = nn.Linear(128, output_size)

    def forward(self, x: Tensor, x_object: Tensor) -> Tensor:
        mu, logvar = self.vae.encoder(x_object)
        template_weights = self.vae.reparametrize(mu, logvar)
        template_weights = template_weights[0].repeat(3, 1, 1, 1)
        template_weights = template_weights.permute(1, 0, 2, 3)
        self.resnet_model.conv1.weight = torch.nn.Parameter(template_weights, requires_grad=False)

        x = self.resnet_model(x)

        x_object = self.resnet_model(x_object)

        x_joined = torch.abs(x - x_object)
        x_joined = self.output(x_joined)
        return x_joined

    def load_vae(self, path: str):
        self.vae.load_state_dict(torch.load(path))
