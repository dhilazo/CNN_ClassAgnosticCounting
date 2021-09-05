import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple

from datasets.spatial_density_counting_dataset import SpatialDensityCountingDataset
from utils.system import join_path, list_files


class ILSVRC(SpatialDensityCountingDataset):
    """
    The ILSVRC dataset is a dataset from ImageNet2015 which contains video frames and the annotation of the bounding
     boxes contained in it.

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

    def __init__(self, data_percentage: float = 1, **kwargs):
        super(ILSVRC, self).__init__(parser=self.get_bounding_box_from_xml, **kwargs)

        subset_folder = "train" if self.train else "val"
        self.data_root = join_path(self.root, "Data", "VID", subset_folder)
        self.annotation_root = join_path(self.root, "Annotations", "VID", subset_folder)

        self.folder_size = []
        for folder in list_files(self.data_root):
            images = list_files(join_path(self.data_root, folder))
            self.size += len(images)
            self.folder_size.append(self.size)

        self.size = int(self.size * data_percentage)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        i = 0
        last_length = 0
        for i, files_length in enumerate(self.folder_size):
            if index < files_length:
                index -= last_length
                break
            last_length = files_length

        folder_image_path = join_path(self.data_root, list_files(self.data_root)[i])
        folder_annotation_path = join_path(self.annotation_root, list_files(self.data_root)[i])

        image_file_path = join_path(folder_image_path, list_files(folder_image_path)[index])
        annotation_file_path = join_path(folder_annotation_path, list_files(folder_annotation_path)[index])

        return self.get_item(image_file_path, annotation_file_path)

    @staticmethod
    def get_bounding_box_from_xml(filename: str) -> Tuple[List, List]:
        bounding_boxes = []
        centers = []

        mytree = ET.parse(filename)
        root = mytree.getroot()
        for box in root.iter("object"):
            xmax = int(box.find("bndbox").find("xmax").text)
            xmin = int(box.find("bndbox").find("xmin").text)
            ymax = int(box.find("bndbox").find("ymax").text)
            ymin = int(box.find("bndbox").find("ymin").text)
            bounding_boxes.append([(xmin, ymin), (xmax, ymax)])
            centers.append(((xmax + xmin) / 2, (ymax + ymin) / 2))
        return bounding_boxes, centers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show random ILSVRC instance")
    parser.add_argument("-p", "--path", type=str, default="../../data/ILSVRC/ILSVRC2015", help="Dataset root path")
    args = parser.parse_args()

    dataset = ILSVRC(root=args.path, image_shape=(255, 255), train=True)

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
