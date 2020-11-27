import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from datasets.ilsvrc_dataset import ILSVRC
from models.cnn_vae import ConvVAE
from models.gmn import GenericMatchingNetwork

if __name__ == "__main__":
    image_shape = (255, 255)
    data_root = './data/ILSVRC/ILSVRC2015'
    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_set = ILSVRC(data_root, image_shape=image_shape, train=False, transform=transform)

    network_model = GenericMatchingNetwork
    model = network_model(output_matching_size=(255 // 4, 255 // 4))
    model.load_state_dict(torch.load("./trained_models/GMN_batch.pt"))
    # model = model.to(device)
    model.eval()

    image, template, ground_truth, count  = val_set[1]
    images = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    templates = torch.reshape(template, (1, template.shape[-3], template.shape[-2], template.shape[-1]))

    outputs = model(images, templates)

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

    axs[2].imshow(ground_truth.numpy()[0])
    axs[2].set_axis_off()
    axs[2].set_title('Ground truth')

    axs[3].imshow(outputs[0].detach().numpy()[0])
    axs[3].set_axis_off()
    axs[3].set_title('Prediction')

    plt.suptitle('GMN')
    plt.show()
