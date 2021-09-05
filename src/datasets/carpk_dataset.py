import argparse
from typing import List, Tuple

from datasets.spatial_density_counting_dataset import SpatialDensityCountingDataset
from utils.system import join_path


class CARPK(SpatialDensityCountingDataset):
    """
    The CARPK dataset is an annotated dataset which contains drone captured images from parking lots. It offers the
    train, val and test splits for the images and annotation.

    Args:
        - root (string): Root directory of dataset.
        - image_shape (Tuple): Tuple containing the desired image size. If the image is larger, it will be resized.
        - train (boolean): Defines to get the train or validation data.
        - transform (callable): transformation function for the images.
    Attributes:
        - data_root (string): Image data root path.
        - annotation_root (string): Annotation data root path.
        - image_shape (Tuple): Where image_shape arg is stored.
        - size (int): Length of the dataset.
    """

    def __init__(self, **kwargs):
        super(CARPK, self).__init__(parser=self.get_bounding_box_from_txt, **kwargs)

        self.subset_folder = "train" if self.train else "test"
        self.image_root = join_path(self.root, "Images")
        self.annotation_root = join_path(self.root, "Annotations")
        self.image_sets_file = join_path(self.root, "ImageSets", self.subset_folder + ".txt")

        self.files = []
        with open(self.image_sets_file, "r") as image_sets_file:
            self.files = image_sets_file.read().splitlines()

        self.size = len(self.files)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        file = self.files[index]

        image_file_path = join_path(self.image_root, file + ".png")
        annotation_file_path = join_path(self.annotation_root, file + ".txt")

        return self.get_item(image_file_path, annotation_file_path)

    @staticmethod
    def get_bounding_box_from_txt(file_path: str) -> Tuple[List, List]:
        bounding_boxes = []
        centers = []

        with open(file_path, "r") as anno_file:
            lines = anno_file.readlines()

            for line in lines:
                xmin, ymin, xmax, ymax = line.split()[:-1]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                bounding_boxes.append([(xmin, ymin), (xmax, ymax)])
                centers.append(((xmax + xmin) / 2, (ymax + ymin) / 2))

        return bounding_boxes, centers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show random CARPK instance")
    parser.add_argument("-p", "--path", type=str, default="../../data/CARPK/CARPK", help="Dataset root path")
    args = parser.parse_args()

    dataset = CARPK(root=args.path, image_shape=(255, 255), train=True)

    import random

    import matplotlib.pyplot as plt

    instance_idx = random.randint(0, len(dataset) - 1)
    input_img, template, ground_truth, object_count, resized_template = dataset[instance_idx]

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(f"Instance #{instance_idx} has {object_count} objects")

    axs[0, 0].set_title("Input")
    axs[0, 0].imshow(input_img)

    axs[0, 1].set_title("Ground truth")
    axs[0, 1].imshow(ground_truth, cmap="gray")

    axs[1, 0].set_title("Template")
    axs[1, 0].imshow(template)

    axs[1, 1].set_title("Resized template")
    axs[1, 1].imshow(resized_template)

    plt.show()
