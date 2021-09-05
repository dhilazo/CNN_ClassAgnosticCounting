import argparse
from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms
from torch import load
from torchvision import datasets

from utils.image import create_image_grid, open_image, pad_image, repeat_image, resize_image
from utils.system import join_path


class CIFAR10CountDataset(datasets.CIFAR10):
    """
    The CIFAR10CountDataset dataset contains grid images of multiple CIFAR10 examples, as well as the counts
    for each class inside the image grid.
    A sample from this dataset will return the image grid, a template for each of the classes contained in the image,
    their appearance object_count and their class labels.

    Args:
        - root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        - image_grid_distribution (tuple): Tuple indicating the image distribution in the grid.
            E.g, (3,3) indicating a 3x3 grid of concatenated images
        - template_view (string, optional): can take values:
            - 'resize': resizess the templates to the same size as the image grid
            - 'padding'(default): sets the template in the center and adds padding around.
            - 'repeat': repeats the template through the image space like a mosaic
        - train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        - transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.

    Attributes:
        - root (string): Where root arg is stored.
        - transform (callable): Where transform arg is stored.
        - n_classes (int): Indicates the number of classes contained by the dataset.
        - image_grid_distribution (tuple): Where image_grid_distribution arg is stored.
        - image_grid_shape (tuple): Tuple indicating the image grid shape.
        - images_per_grid (int): Number of images contained by each image grid.
        - template_dict (dictionary): Contains a template image for each class name.
        - class_names (list): List of class names for each index label.
        - resize_template (string): Where resize_template arg is stored.
    """

    def __init__(
        self,
        root: str = "",
        image_grid_distribution: Tuple[int, int] = (3, 3),
        template_view: str = "padding",
        train: bool = True,
        transform: transforms.Compose = None,
    ):
        super().__init__(root, train, download=True)
        self.root = root
        self.transform = transform
        self.n_classes = 10
        self.image_grid_distribution = image_grid_distribution
        self.image_grid_shape = (3, image_grid_distribution[0] * 96, image_grid_distribution[1] * 96)
        self.images_per_grid = np.prod(self.image_grid_distribution)
        self.class_names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        self.template_dict = self.create_template_dict(self.class_names)
        self.template_view = template_view

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: int):
        indices = np.arange(index, index - self.images_per_grid, -1)

        images = []
        labels = []
        for index in indices:
            image, label = super().__getitem__(index)
            image = resize_image(image, (96, 96))
            images.append(image)
            labels.append(label)

        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        counts = np.zeros(self.n_classes, dtype=np.float32)
        counts[unique_labels] = unique_counts
        counts = counts.reshape((counts.shape[0], 1))

        image_grid = create_image_grid(images, self.image_grid_distribution)

        if self.transform is not None:
            image_grid = self.transform(image_grid)

        templates = []
        for class_name in self.class_names:
            template = self.template_dict[class_name]
            template = resize_image(template, (96, 96))
            # Make template have the same shape as the image grid
            if self.template_view == "resize":
                template = resize_image(template, self.image_grid_shape[-2:])
            elif self.template_view == "padding":
                template = pad_image(template, self.image_grid_shape[-2:])
            elif self.template_view == "repeat":
                template = repeat_image(template, self.image_grid_shape[-2:])

            if self.transform is not None:
                template = self.transform(template)

            templates.append(template)

        return image_grid, templates, counts

    def create_template_dict(self, classes: Sequence, to_tensor: bool = False) -> dict:
        template_dict = {}
        for class_name in classes:
            if to_tensor:
                template_dict[class_name] = load(join_path(self.root, f"templates/{class_name}.pt"))
            else:
                template_dict[class_name] = open_image(join_path(self.root, f"templates/{class_name}.jpg"))
        return template_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show random CIFAR10 Count instance")
    parser.add_argument("-p", "--path", type=str, default="../../data/CIFAR10", help="Dataset root path")
    args = parser.parse_args()

    dataset = CIFAR10CountDataset(root=args.path, image_grid_distribution=(3, 3), template_view="raw", train=True)

    import random

    import matplotlib.pyplot as plt

    instance_idx = random.randint(0, len(dataset) - 1)
    input_img, templates, ground_truth_counts = dataset[instance_idx]

    template_idx = random.randint(0, dataset.n_classes - 1)
    template, object_count = templates[template_idx], ground_truth_counts[template_idx][0]

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    plt.suptitle(f"Instance #{instance_idx} has {int(object_count)} {dataset.class_names[template_idx]}/s")

    axs[0].set_title("Input")
    axs[0].imshow(input_img)

    axs[1].set_title("Template")
    axs[1].imshow(template)

    plt.show()
