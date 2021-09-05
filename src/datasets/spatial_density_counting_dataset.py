from abc import ABC
from typing import Callable, Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset

from utils.image import add_square_padding, paste_image, resize_image, thumbnail_image


def get_gaussian_kernel(shape: Sequence) -> np.ndarray:
    """Returns an N-Dimensional Gaussian kernel array based on given shape."""
    kernel_list = [signal.gaussian(axis, std=axis / 3.5).reshape(axis, 1) for axis in shape]
    gaussian_kernel = np.outer(*kernel_list)
    return gaussian_kernel


class SpatialDensityCountingDataset(ABC, Dataset):
    """
    Abstract class for annotated Spatial Density Counting datasets.

    Args:
        - root (string): Root directory of dataset.
        - image_shape (Tuple): Tuple containing the desired image size. If the image is larger, it will be resized.
        - train (boolean): Defines to get the train or validation data.
        - transform (callable): transformation function for the images.
    Attributes:
        - image_shape (Tuple): Where image_shape arg is stored.
        - size (int): Length of the dataset.
    """

    class Annotation:
        """
        Dataset's bounding box annotation.

        Args:
            - root (string): Path to the annotation file extracted from the dataset.
        Attributes:
            - bounding_boxes (List): List containing the corners of the squared annotation.
            - centers (List): List which contains the center position of each annotation.
        """

        def __init__(self, file_path: str, parser: Callable):
            self.bounding_boxes = []
            self.centers = []

            if parser:
                self.bounding_boxes, self.centers = parser(file_path)

    def __init__(
        self,
        parser: Callable = None,
        root: str = ".",
        image_shape: Tuple[int, int] = None,
        train: bool = True,
        transform: transforms.Compose = None,
    ):
        self.annotation_parser = parser
        self.root = root
        self.image_shape = image_shape
        self.train = train
        self.transform = transform
        self.size = 0

    def __len__(self) -> int:
        raise NotImplementedError

    def get_item(self, image_file_path: str, annotation_file_path: str):
        im = Image.open(image_file_path)
        annotation = self.Annotation(annotation_file_path, self.annotation_parser)

        if self.image_shape is None:
            ground_truth_size = im.size
        else:
            ground_truth_size = self.image_shape

        biggest_bounding_box, ground_truth = self.get_ground_truth_image(annotation, im, ground_truth_size)

        if biggest_bounding_box is not None:
            coords = self.get_squared_bbox(biggest_bounding_box, im.size)

            template = im.crop(tuple(coords))
            template = thumbnail_image(template, (63, 63))
        else:
            template = Image.new("RGB", (63, 63))

        if self.image_shape is not None:
            im = thumbnail_image(im, self.image_shape)
            im = add_square_padding(im, self.image_shape)

            ground_truth_size = (self.image_shape[0] // 4, self.image_shape[1] // 4)
            ground_truth = thumbnail_image(ground_truth, ground_truth_size)
            ground_truth = add_square_padding(ground_truth, ground_truth_size, output_mode="L")

        object_count = len(annotation.centers)
        resized_template = resize_image(template, (96, 96))

        if self.transform is not None:
            im = self.transform(im)
            template = self.transform(template)
            ground_truth = self.transform(ground_truth)
            resized_template = self.transform(resized_template)

        return im, template, ground_truth, object_count, resized_template

    @staticmethod
    def get_ground_truth_image(
        annotation: Annotation, im: Image.Image, size: Tuple[int, int]
    ) -> Tuple[Sequence, Image.Image]:
        ground_truth = Image.new("L", size)
        biggest_bounding_box = None
        max_area = 0
        for annotation_center, bounding_box in zip(annotation.centers, annotation.bounding_boxes):
            current_area = (bounding_box[1][0] - bounding_box[0][0]) * (bounding_box[1][1] - bounding_box[0][1])
            if max_area < current_area:
                max_area = (bounding_box[1][0] - bounding_box[0][0]) * (bounding_box[1][1] - bounding_box[0][1])
                biggest_bounding_box = bounding_box

            rescaled_bbox = [
                [int(bounding_box[0][0] * size[0] / im.size[0]), int(bounding_box[0][1] * size[1] / im.size[1])],
                [int(bounding_box[1][0] * size[0] / im.size[0]), int(bounding_box[1][1] * size[1] / im.size[1])],
            ]
            shape = [(rescaled_bbox[1][0] - rescaled_bbox[0][0]), (rescaled_bbox[1][1] - rescaled_bbox[0][1])]

            annotation_center = (
                int((annotation_center[0] * size[0] / im.size[0]) - shape[0] // 2),
                int((annotation_center[1] * size[1] / im.size[1]) - shape[1] // 2),
            )

            gaussian = get_gaussian_kernel(shape)
            gaussian *= 255
            gaussian = gaussian.astype(np.uint8)
            gaussian = Image.fromarray(gaussian, mode="L")

            ground_truth = paste_image(ground_truth, gaussian, annotation_center)

        return biggest_bounding_box, ground_truth

    @staticmethod
    def get_squared_bbox(template_box: Sequence, image_shape: Tuple[int, int]) -> Sequence:
        x_dist = template_box[1][0] - template_box[0][0]
        y_dist = template_box[1][1] - template_box[0][1]

        distances = [x_dist, y_dist]
        max_axis = np.argmax(distances)
        max_distance = max(distances[max_axis], 63)

        coords = [template_box[0][0], template_box[0][1], template_box[1][0], template_box[1][1]]

        coords[0] -= (max_distance - x_dist) // 2
        coords[1] -= (max_distance - y_dist) // 2

        coords[2] += ((max_distance - x_dist) // 2) + 1 * ((max_distance - x_dist) % 2)
        coords[3] += ((max_distance - y_dist) // 2) + 1 * ((max_distance - y_dist) % 2)

        # Ensure frame is inside picture
        old_coords = coords.copy()
        coords[0] = max(0, old_coords[0] - max(0, old_coords[2] - image_shape[0]))
        coords[1] = max(0, old_coords[1] - max(0, old_coords[3] - image_shape[1]))

        coords[2] = min(image_shape[0], old_coords[2] + max(0, -old_coords[0]))
        coords[3] = min(image_shape[1], old_coords[3] + max(0, -old_coords[1]))

        return coords
