import torch
import torch.nn.functional as F
from torch import nn


class SiameseNet(nn.Module):
    def __init__(self, channels=3, output_size=10):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x, x_object):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_object = self.pool(F.relu(self.conv1(x_object)))
        x_object = self.pool(F.relu(self.conv2(x_object)))
        x_object = x_object.view(-1, 16 * 21 * 21)
        x_object = F.relu(self.fc1(x_object))
        x_object = F.relu(self.fc2(x_object))

        x_joined = torch.abs(x - x_object)
        x_joined = self.fc3(x_joined)
        return x_joined
