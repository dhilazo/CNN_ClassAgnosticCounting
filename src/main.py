import os

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split, DataLoader

from CIFAR10CountDataset import CIFAR10CountDataset
from Trainer import Trainer
from models.siamese_model import SiameseNet


def save_template(train_loader, classes):
    found = []
    # get some random training images
    dataiter = iter(train_loader)
    while len(found) != len(classes):
        images, labels = dataiter.next()
        for img, label in zip(images, labels):
            if label not in found:
                found.append(label)
                # Save and remove
                torch.save(img, './data/templates/' + classes[label] + '.pt')


def create_log_dirs(run_name):
    path = os.path.join('.', 'logs', run_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_template_dict(classes):
    template_dict = {}
    for class_name in classes:
        template_dict[class_name] = torch.load(f'./data/templates/{class_name}.pt')

    return template_dict


def create_grid(inputs, num_images, image_grid_shape, labels):
    inputs = torch.reshape(inputs, (num_images, -1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]))
    labels = torch.reshape(labels, (num_images, -1))
    new_inputs = torch.zeros(
        (num_images, 3, inputs.shape[-2] * image_grid_shape[0], inputs.shape[-1] * image_grid_shape[1]))
    for i, input in enumerate(inputs):
        rows = []
        for row in range(image_grid_shape[0]):
            index = row * image_grid_shape[1]
            image_row = torch.cat((input[index], input[index + 1], input[index + 2]), -1)
            rows.append(image_row)
        image_grid = torch.cat(tuple(rows), -2)
        new_inputs[i] = image_grid
        # imshow(new_inputs[i])
        # plt.show()
    return new_inputs, labels


if __name__ == "__main__":
    run_name = 'SiameseNet_Count_Pad'
    network_model = SiameseNet
    epochs = 100
    image_grid_distribution = (3, 3)
    batch_size = 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10CountDataset('./data', image_grid_distribution, train=True, transformations=transform)
    test_set = CIFAR10CountDataset('./data', image_grid_distribution, train=False, transformations=transform)

    train_len = len(train_set)
    train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = network_model(output_size=1)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainer = Trainer(model, criterion, optimizer, run_name, device=device)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')

    trainer.evaluate(test_loader)
