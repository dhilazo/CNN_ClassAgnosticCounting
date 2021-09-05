import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50


class FullyConnectedModel(nn.Module):
    def __init__(self, output_size: int):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class DoubleInputNet(nn.Module):
    def __init__(self, channels: int = 3, output_size: int = 10):
        super(DoubleInputNet, self).__init__()
        self.obj_conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = resnet50(pretrained=True)
        self.model.fc = FullyConnectedModel(output_size)

    def forward(self, x: Tensor, x_object: Tensor) -> Tensor:
        x_object = self.obj_conv1(x_object)
        self.model.conv1.weight = self.obj_conv1.weight
        x = self.model(x)
        return x
