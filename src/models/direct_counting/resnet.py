import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet50


class FullyConnectedModel(nn.Module):
    def __init__(self, output_size: int):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class ResNet(nn.Module):
    def __init__(self, output_size: int = 10):
        super(ResNet, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = FullyConnectedModel(output_size)

    def forward(self, x: Tensor, x_object: Tensor) -> Tensor:
        x = self.model(x)
        return x
