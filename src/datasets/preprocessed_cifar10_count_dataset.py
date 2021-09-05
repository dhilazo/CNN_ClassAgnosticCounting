import argparse
from typing import Callable, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.system import join_path, list_files


class PreprocessedCIFAR10CountDataset(Dataset):
    """
    This dataset is extracted from the CIFAR10CountDataset by running the save_preprocessed_cifar_dataset.py script.

    The CIFAR10CountDataset dataset contains grid images of multiple CIFAR10 examples, as well as the counts
    for each class inside the image grid.
    A sample from this dataset will return the image grid, a template for each of the classes contained in the image,
    their appearance object_count and their class labels.

    Args:
        - root (string): Root directory of dataset.
        - train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        - transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.

    Attributes:
        - root (string): Where root arg is stored.
        - class_names (list): List of class names for each index label.
        - size(int): Length of the dataset
        - resize_template (string): Where resize_template arg is stored.
        - transform (callable): Where transform arg is stored.

    """

    def __init__(self, root: str, train: bool = True, transform: Callable = None):
        self.root = root
        self.class_names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        self.templates_path = join_path(self.root, "templates")
        self.transform = transform
        self.root = join_path(self.root, "train") if train else join_path(self.root, "test")
        self.files_list = list_files(join_path(self.root, "images"))
        self.size = len(self.files_list)

        with open(join_path(self.root, "counts", "counts.txt"), "r") as counts_file:
            self.count_lines = counts_file.readlines()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        im = Image.open(join_path(self.root, "images", self.files_list[index]))
        templates = self.open_templates()
        counts = self.count_lines[index].split(" ")
        counts = np.asarray([int(string) for string in counts], dtype=np.float32)
        counts = counts.reshape((counts.shape[0], 1))

        if self.transform is not None:
            im = self.transform(im)

        return im, templates, counts

    def open_templates(self) -> List:
        templates = []
        for class_name in self.class_names:
            template_name = f"{class_name}.jpg"
            template = Image.open(join_path(self.templates_path, template_name))
            if self.transform is not None:
                template = self.transform(template)
            templates.append(template)
        return templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show random CIFAR10 Count instance")
    parser.add_argument("-p", "--path", type=str, default="../../data/CIFAR10Count", help="Dataset root path")
    args = parser.parse_args()

    dataset = PreprocessedCIFAR10CountDataset(root=args.path, train=True)

    import random

    import matplotlib.pyplot as plt

    instance_idx = random.randint(0, len(dataset) - 1)
    input_img, templates, ground_truth_counts = dataset[instance_idx]

    template_idx = random.randint(0, len(dataset.class_names) - 1)
    template, object_count = templates[template_idx], ground_truth_counts[template_idx][0]

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    plt.suptitle(f"Instance #{instance_idx} has {int(object_count)} {dataset.class_names[template_idx]}/s")

    axs[0].set_title("Input")
    axs[0].imshow(input_img)

    axs[1].set_title("Template")
    axs[1].imshow(template)

    plt.show()
