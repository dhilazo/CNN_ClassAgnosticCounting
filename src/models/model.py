import torch.nn as nn
import torch.nn.functional as F


class DoubleInputNet(nn.Module):
    def __init__(self, channels=3, output_size=10):
        super(DoubleInputNet, self).__init__()
        self.obj_conv1 = nn.Conv2d(channels, 6, 5)

        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x, x_object):
        x_object = self.obj_conv1(x_object)
        self.conv1.weight = self.obj_conv1.weight
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = DoubleInputNet()
