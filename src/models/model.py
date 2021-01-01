import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class FullyConnectedModel(nn.Module):
    def __init__(self, output_size):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class DoubleInputNet(nn.Module):
    def __init__(self, channels=3, output_size=10):
        super(DoubleInputNet, self).__init__()
        self.obj_conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = resnet50(pretrained=True)
        self.model.fc = FullyConnectedModel(output_size)

    def forward(self, x, x_object):
        x_object = self.obj_conv1(x_object)
        self.model.conv1.weight = self.obj_conv1.weight
        x = self.model(x)
        return x

if __name__ == '__main__':
    model = DoubleInputNet(output_size=1)
    data = torch.zeros(2, 3, 288, 288)
    template = torch.zeros(2, 3, 96, 96)
    assert model(data, template)[0].shape == 1
