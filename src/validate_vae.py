import argparse
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from datasets import SpatialDensityCountingDataset, dataset_dict
from models import vae_model_dict
from utils.decorator import counting_script


@counting_script
def validate_vae(parser: Optional[argparse.ArgumentParser] = None):
    parser = argparse.ArgumentParser(
        description="Plots the output density map of a trained counting model.", parents=[parser]
    )
    parser.add_argument(
        "-v",
        "--vae-model",
        type=str,
        default="ConvVAE",
        choices=vae_model_dict.keys(),
        help="VAE model to validate",
    )
    parser.add_argument(
        "-vp", "--vae-path", type=str, default="../trained_models/ConvVAE.pt", help="Path to pretrained VAE weights"
    )
    args = parser.parse_args()

    network_model = vae_model_dict[args.vae_model]
    model_name = network_model.__name__
    dataset = dataset_dict[args.dataset]
    image_shape = (args.image_shape, args.image_shape)
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    if issubclass(dataset, SpatialDensityCountingDataset):
        kwargs = dict(root=args.data_path, image_shape=image_shape, transform=transform)
    else:
        kwargs = dict(root=args.data_path, transform=transform)
    test_set = dataset(train=False, **kwargs)

    model = network_model()
    model.load_state_dict(torch.load(args.vae_path, map_location=device))
    model.eval()

    index = random.randint(0, len(test_set))
    if issubclass(dataset, SpatialDensityCountingDataset):
        _, _, _, _, resized_template = test_set[index]
        template = torch.reshape(resized_template, (1, *resized_template.shape[-3:]))
    else:
        _, templates, _ = test_set[index]
        template = torch.reshape(templates[0], (1, *templates[0].shape[-3:]))

    decoded, mu, _ = model(template)

    template = np.moveaxis(template[0].detach().numpy(), 0, -1)
    decoded = np.moveaxis(decoded[0].detach().numpy(), 0, -1)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 8), facecolor="w", edgecolor="k")

    axs = axs.ravel()
    plt.axis("off")

    axs[0].imshow(template)
    axs[0].set_axis_off()
    axs[0].set_title("Original image")

    axs[1].imshow(decoded)
    axs[1].set_axis_off()
    axs[1].set_title("Decoded")

    plt.suptitle(f"{model_name}")

    plt.show()


if __name__ == "__main__":
    validate_vae()
