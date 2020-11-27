import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset

from utils.image import paste_image, thumbnail_image
from utils.system import join_path, list_files


def gkern(shape):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_x = signal.gaussian(shape[0], std=shape[0] / 3.5).reshape(shape[0], 1)
    gkern1d_y = signal.gaussian(shape[1], std=shape[1] / 3.5).reshape(shape[1], 1)

    gkern2d = np.outer(gkern1d_y, gkern1d_x)
    return gkern2d


class Annotation:
    def __init__(self, filename):
        self.bounding_boxes = []
        self.centers = []
        self.get_bounding_box_from_xml(filename)

    def get_bounding_box_from_xml(self, filename):
        mytree = ET.parse(filename)
        root = mytree.getroot()
        for box in root.iter('object'):
            xmax = int(box.find('bndbox').find('xmax').text)
            xmin = int(box.find('bndbox').find('xmin').text)
            ymax = int(box.find('bndbox').find('ymax').text)
            ymin = int(box.find('bndbox').find('ymin').text)
            self.bounding_boxes.append([(xmin, ymin), (xmax, ymax)])
            self.centers.append(((xmax + xmin) / 2, (ymax + ymin) / 2))


class ILSVRC(Dataset):
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
        - anno_root (string): Annotation data root path.
        - image_shape (Tuple): Where image_shape arg is stored.
        - size (int): Length of the dataset.
    """

    def __init__(self, root, image_shape=None, data_percentage=1, train=True, transform=None):
        self.subset_folder = 'train' if train else 'val'
        self.data_root = join_path(root, 'Data', 'VID', self.subset_folder)
        self.anno_root = join_path(root, 'Annotations', 'VID', self.subset_folder)
        self.image_shape = image_shape
        self.transform = transform
        self.size = 0
        self.folder_size = []
        for folder in list_files(self.data_root):
            images = list_files(join_path(self.data_root, folder))
            self.size += len(images)
            self.folder_size.append(self.size)
        self.size = int(self.size * data_percentage)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        i = 0
        last_length = 0
        for i, files_length in enumerate(self.folder_size):
            if index < files_length:
                index -= last_length
                break
            last_length = files_length

        folder_image_path = join_path(self.data_root, list_files(self.data_root)[i])
        folder_anno_path = join_path(self.anno_root, list_files(self.data_root)[i])

        im = Image.open(join_path(folder_image_path, list_files(folder_image_path)[index]))
        annotations = Annotation(join_path(folder_anno_path, list_files(folder_anno_path)[index]))
        if self.image_shape is None:
            size = im.size
        else:
            size = self.image_shape
        size = (size[0] // 4, size[1] // 4)
        ground_truth = Image.new('L', size)
        template_box = None
        max_area = 0
        for gaussian_center, bounding_box in zip(annotations.centers, annotations.bounding_boxes):
            current_area = ((bounding_box[1][0] - bounding_box[0][0]) * (bounding_box[1][1] - bounding_box[0][1]))
            if max_area < current_area:
                max_area = ((bounding_box[1][0] - bounding_box[0][0]) * (bounding_box[1][1] - bounding_box[0][1]))
                template_box = bounding_box
            box = [[int(bounding_box[0][0] * self.image_shape[0] / im.size[0]),
                    int(bounding_box[0][1] * self.image_shape[1] / im.size[1])],
                   [int(bounding_box[1][0] * self.image_shape[0] / im.size[0]),
                    int(bounding_box[1][1] * self.image_shape[1] / im.size[1])]]
            shape = [(box[1][0] - box[0][0]) // 4, (box[1][1] - box[0][1]) // 4]
            gaussian = gkern(shape)
            gaussian *= 255
            gaussian = gaussian.astype(np.uint8)
            gaussian = Image.fromarray(gaussian, mode='L')
            gaussian_center = (
            int((gaussian_center[0] * self.image_shape[0] / im.size[0]) // 4 - gaussian.size[0] // 2),
            int((gaussian_center[1] * self.image_shape[1] / im.size[1]) // 4 - gaussian.size[1] // 2))
            ground_truth = paste_image(ground_truth, gaussian, gaussian_center)
        # ground_truth.show()
        if template_box is not None:
            coords = self.square_template(template_box, im)

            template = im.crop(tuple(coords))
            template = thumbnail_image(template, (63, 63))
        else:
            template = Image.new('RGB', (63, 63))

        if self.image_shape is not None:
            im = thumbnail_image(im, self.image_shape)
            padding = Image.new('RGBA', self.image_shape)
            x = int(padding.size[0] / 2 - im.size[0] / 2)
            y = int(padding.size[1] / 2 - im.size[1] / 2)
            im = im.convert('RGBA')
            im = paste_image(padding, im, (x, y)).convert('RGB')
        # im.show()
        # template.show()
        count = len(annotations.centers)

        if self.transform is not None:
            im = self.transform(im)
            template = self.transform(template)
            ground_truth = self.transform(ground_truth)

        return im, template, ground_truth, count

    @staticmethod
    def square_template(template_box, im):
        x_dist = (template_box[1][0] - template_box[0][0])
        y_dist = (template_box[1][1] - template_box[0][1])

        distances = [x_dist, y_dist]
        max_axis = np.argmax(distances)

        max = distances[max_axis]

        coords = [template_box[0][0], template_box[0][1], template_box[1][0], template_box[1][1]]

        if max < 63:
            coords[0] -= ((63 - x_dist) // 2) + 1
            coords[2] += ((63 - x_dist) // 2) + 1 * ((63 - x_dist) % 2)

            coords[1] -= ((63 - y_dist) // 2) + 1
            coords[3] += ((63 - y_dist) // 2) + 1 * ((63 - y_dist) % 2)
        else:
            if max_axis == 0:
                add = (x_dist - y_dist) // 2
                coords[1] -= add
                coords[3] += add
                if coords[3] - coords[1] != x_dist:
                    coords[3] += 1
            if max_axis == 1:
                add = (y_dist - x_dist) // 2
                coords[0] -= add
                coords[2] += add
                if coords[2] - coords[0] != y_dist:
                    coords[2] += 1

        # Check inside picture
        for i in range(len(coords)):
            if i < 2 and coords[i] < 0:
                coords[i + 2] += 0 - coords[i]
                coords[i] = 0
            if i >= 2 and coords[i] > im.size[i % 2]:
                coords[i - 2] -= coords[i] - im.size[i % 2]
                coords[i] = im.size[i % 2]
        return coords
