import torch
import torch.nn.functional as F
from torch import nn


class SiameseNet(nn.Module):
    def __init__(self, channels=3, output_size=10):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*21*21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x, x_object):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*21*21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_object = self.pool(F.relu(self.conv1(x_object)))
        x_object = self.pool(F.relu(self.conv2(x_object)))
        x_object = x_object.view(-1, 16*21*21)
        x_object = F.relu(self.fc1(x_object))
        x_object = F.relu(self.fc2(x_object))

        x_joined = torch.abs(x - x_object)
        x_joined = self.fc3(x_joined)
        return x_joined

    # def __init__(self, channels=3, output_size=10):
    #     super(SiameseNet, self).__init__()
    #
    #     # Conv2d(input_channels, output_channels, kernel_size)
    #     self.conv1 = nn.Conv2d(channels, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #
    #     self.bn1 = nn.BatchNorm2d(6)
    #     self.bn2 = nn.BatchNorm2d(16)
    #     self.dropout1 = nn.Dropout(0.1)
    #     self.dropout2 = nn.Dropout(0.5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fcOut = nn.Linear(120, output_size)
    #
    # def convs(self, x):
    #     # out_dim = in_dim - kernel_size + 1
    #     # Batch images with 3 channels each with 32x32 pixels
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     # 64, 23x23
    #     x = F.max_pool2d(x, (2, 2))
    #     # 64, 11x11
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     # 128, 5x5
    #     x = F.max_pool2d(x, (2, 2))
    #
    #     return x
    #
    # def forward(self, x1, x2):
    #     x1 = self.convs(x1)
    #     x1 = x1.view(-1, 16 * 5 * 5)
    #     x1 = F.relu(self.fc1(x1))
    #
    #     x2 = self.convs(x2)
    #     x2 = x2.view(-1, 16 * 5 * 5)
    #     x2 = F.relu(self.fc1(x2))
    #
    #     x = torch.abs(x1 - x2)
    #     x = self.fcOut(x)
    #     return x
