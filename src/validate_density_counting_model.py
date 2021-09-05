import argparse
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

from datasets import SpatialDensityCountingDataset, dataset_dict
from models import counting_model_dict
from models.density_counting import density_counting_models
from utils.count import count_local_maximums
from utils.decorator import counting_script


@counting_script
def validate_density_counting_model(parser: Optional[argparse.ArgumentParser] = None):
    parser = argparse.ArgumentParser(
        description="Plots the output density map of a trained counting model.", parents=[parser]
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="SiameseGMN",
        choices=counting_model_dict.keys(),
        help="Counting model",
    )
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=str,
        default="../trained_models/SiameseGMN_batch.pt",
        help="Path to trained weights",
    )
    args = parser.parse_args()

    network_model = counting_model_dict[args.model]
    model_name = network_model.__name__
    dataset = dataset_dict[args.dataset]
    image_shape = (args.image_shape, args.image_shape)
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    if issubclass(dataset, SpatialDensityCountingDataset):
        kwargs = dict(root=args.data_path, image_shape=image_shape, transform=transform)
    else:
        kwargs = dict(root=args.data_path, transform=transform)
    val_set = dataset(train=False, **kwargs)

    if network_model not in density_counting_models:
        raise ValueError("Output from non-density counting models can't be plotted")

    model = network_model(output_matching_size=(image_shape[0] // 4, image_shape[1] // 4))

    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    except RuntimeError:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
        if device == torch.device("cpu"):
            model = model.module

    model = model.to(device)

    index = random.randint(0, len(val_set))
    model.eval()
    image, template, ground_truth, count, resized_template = val_set[index]
    images = torch.reshape(image, (1, *image.shape[-3:]))
    templates = torch.reshape(template, (1, *template.shape[-3:]))
    resized_template = torch.reshape(resized_template, (1, *resized_template.shape[-3:]))

    outputs = model(images, templates, resized_template)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor="w", edgecolor="k")

    axs = axs.ravel()
    plt.axis("off")

    axs[0].imshow(np.moveaxis(image.numpy(), 0, -1))
    axs[0].set_axis_off()
    axs[0].set_title("Original image")

    axs[1].imshow(np.moveaxis(template.numpy(), 0, -1))
    axs[1].set_axis_off()
    axs[1].set_title("Template")

    axs[2].imshow(ground_truth.numpy()[0], cmap="gray")
    axs[2].set_axis_off()
    axs[2].set_title(f"Ground truth (Count: {count})")

    axs[3].imshow(outputs[0].detach().cpu().numpy()[0], cmap="gray")
    axs[3].set_axis_off()
    axs[3].set_title(f"Prediction (Count: {count_local_maximums(outputs[0].detach().cpu().numpy()[0])})")

    plt.suptitle(f"{model_name}")

    plt.show()


if __name__ == "__main__":
    validate_density_counting_model()
