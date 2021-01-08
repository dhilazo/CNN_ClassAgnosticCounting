import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim

from datasets.carpk_dataset import CARPK
from datasets.ilsvrc_dataset import ILSVRC
from models.adapted_gmn import AdaptedGenericMatchingNetwork
from models.cnn_vae import ConvVAE
from models.gmn import GenericMatchingNetwork
from models.gmn_etcnet import GMNETCNet
from models.siamese_gmn import SiameseGenericMatchingNetwork

if __name__ == "__main__":
    image_shape = (255, 255)
    data_root = './data/ILSVRC/ILSVRC2015'
    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_root = './data/CARPK/CARPK'
    val_set = CARPK(data_root, image_shape=image_shape, train=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    network_model = GMNETCNet
    model = network_model(output_matching_size=(255 // 4, 255 // 4))
    model_name = type(model).__name__
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("../test/trained_models3.0/GMNETCNet_final_batch.pt"))
    model = model.to(device)
    model.train()

    index = random.randint(0, len(val_set))
    index = 812
    print(index)
    image, template, ground_truth, count, resized_template = val_set[index]
    images = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    templates = torch.reshape(template, (1, template.shape[-3], template.shape[-2], template.shape[-1]))
    resized_template = torch.reshape(resized_template,
                                     (1, resized_template.shape[-3], resized_template.shape[-2],
                                      resized_template.shape[-1]))

    outputs = model(images, templates, resized_template)
    # outputs = model(images, templates)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor='w', edgecolor='k')

    axs = axs.ravel()
    plt.axis('off')

    axs[0].imshow(np.moveaxis(image.numpy(), 0, -1))
    axs[0].set_axis_off()
    axs[0].set_title('Original image')

    axs[1].imshow(np.moveaxis(template.numpy(), 0, -1))
    axs[1].set_axis_off()
    axs[1].set_title('Template')

    axs[2].imshow(ground_truth.numpy()[0], cmap="gray")
    axs[2].set_axis_off()
    axs[2].set_title('Ground truth')

    axs[3].imshow(outputs[0].detach().cpu().numpy()[0], cmap="gray")
    axs[3].set_axis_off()
    axs[3].set_title('Prediction')

    plt.suptitle('GMN')

    print(count, SiameseGenericMatchingNetwork.get_count(outputs[0].detach().cpu().numpy()[0], plot=True))
    plt.show()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        train_loss = []

        loss_count = 0
        for i, data in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, templates, ground_truth, count, resized_template = data

            images = images.to(device)
            templates = templates.to(device)
            ground_truth = ground_truth.to(device)
            resized_template = resized_template.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images, templates, resized_template)

            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()
            if i == 3:
                break
    model.eval()
    count = 0
    # while count < 2:
    image, template, ground_truth, count, resized_template = val_set[index]
    images = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    templates = torch.reshape(template, (1, template.shape[-3], template.shape[-2], template.shape[-1]))
    resized_template = torch.reshape(resized_template,
                                     (1, resized_template.shape[-3], resized_template.shape[-2],
                                      resized_template.shape[-1]))

    outputs = model(images, templates, resized_template)
    # outputs = model(images, templates)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor='w', edgecolor='k')

    axs = axs.ravel()
    plt.axis('off')

    axs[0].imshow(np.moveaxis(image.numpy(), 0, -1))
    axs[0].set_axis_off()
    axs[0].set_title('Original image')

    axs[1].imshow(np.moveaxis(template.numpy(), 0, -1))
    axs[1].set_axis_off()
    axs[1].set_title('Template')

    axs[2].imshow(ground_truth.numpy()[0], cmap="gray")
    axs[2].set_axis_off()
    axs[2].set_title('Ground truth')

    axs[3].imshow(outputs[0].detach().cpu().numpy()[0], cmap="gray")
    axs[3].set_axis_off()
    axs[3].set_title('Prediction')

    plt.suptitle('GMN')

    print(count, SiameseGenericMatchingNetwork.get_count(outputs[0].detach().cpu().numpy()[0], plot=True))
    plt.show()

    plt.imsave(f"../{model_name}_CIFAR.jpg", outputs[0].detach().cpu().numpy()[0], cmap="gray")
