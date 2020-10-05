import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # TODO no nn. imports
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from models.model import DoubleInputNet
from models.siamese_model import SiameseNet


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


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


def quick_validate(model, val_loader, criterion, template_dict=None):
    model.eval()
    with torch.no_grad():
        epoch_val_loss = []
        for ii, (features, gt) in enumerate(val_loader):
            if template_dict is not None:
                templates, counts = get_templates_and_counts(template_dict, labels, classes)
                # templates = get_templates(template_dict, labels, classes, template_shape=inputs[0].shape[-2:])
                prediction = model(features, templates)
            else:
                prediction = model(features, features)

            loss = criterion(prediction, gt)

            epoch_val_loss.append(loss.item())
    model.train()
    return epoch_val_loss


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


def get_templates_and_counts(template_dict, labels, classes, template_shape=None):
    template_list = []
    counts_list = []
    for input in labels:
        unique_values, counts = np.unique(input, return_counts=True)
        counts_list.append(counts)

        templates = []
        for value in unique_values:
            if not template_shape:
                templates.append(F.pad(template_dict[classes[value.item()]], pad=[32, 32, 32, 32], mode='constant'))
            else:
                templates.append(
                    F.interpolate(torch.Tensor([template_dict[classes[value.item()]].numpy()]), size=template_shape))
        template_list.append(templates)
    return template_list, counts


if __name__ == "__main__":
    run_name = 'SiameseNet_Count'
    network_model = SiameseNet
    num_epochs = 100
    images = 4
    image_grid_shape = (3, 3)
    batch_size = image_grid_shape[0] * image_grid_shape[1] * images

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_len = len(train_set)
    train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # save_template(train_loader, classes)
    template_dict = create_template_dict(
        classes) if network_model is DoubleInputNet or network_model is SiameseNet else None
    # imshow(torchvision.utils.make_grid(list(template_dict.values())))
    # plt.show()

    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    model = network_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    logs_path = create_log_dirs(run_name)
    train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logs_path, 'val'))

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        train_loss = []
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = create_grid(inputs, images, image_grid_shape, labels)  # TODO continuar
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if template_dict is not None:
                templates, counts = get_templates_and_counts(template_dict, labels, classes)
                # templates = get_templates(template_dict, labels, classes, template_shape=inputs[0].shape[-2:])
                outputs = model(inputs, templates)
            else:
                outputs = model(inputs, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss.append(loss.item())
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        val_loss = quick_validate(model, val_loader, criterion, template_dict=template_dict)

        train_mean_loss = np.mean(train_loss)
        val_mean_loss = np.mean(val_loss)
        print(f'Losses: {train_mean_loss} {val_mean_loss}')

        train_writer.add_scalar('loss', train_mean_loss, epoch + 1)
        val_writer.add_scalar('loss', val_mean_loss, epoch + 1)

    print('Finished Training')

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if template_dict is not None:
                templates = torch.stack([template_dict[classes[label.item()]] for label in labels])
                outputs = model(images, templates)
            else:
                outputs = model(images, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         if template_dict is not None:
    #             templates = torch.stack([template_dict[classes[label.item()]] for label in labels])
    #             outputs = model(images, templates)
    #         else:
    #             outputs = model(images, images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(images):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    #
    # plt.show()
