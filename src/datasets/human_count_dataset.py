from PIL import Image
from torch.utils.data import Dataset

from utils.image import thumbnail_image
from utils.system import join_path, list_files


class HumanCountDataset(Dataset):
    """
    The HumanCountDataset dataset contains surveillance cameras images from shanghai which can be used for crowd
    counting.
    A sample from this dataset will return the image and the amount of people captured by it.

    Args:
        - root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

    Attributes:
        - root (string): Where root arg is stored.
    """

    def __init__(self, root, image_shape=None, train=True, transform=None):
        self.root = root
        self.template_path = join_path(self.root, 'human_template.png')
        self.image_shape = image_shape
        self.transform = transform
        if train:
            self.root = join_path(self.root, 'train_data')
        else:
            self.root = join_path(self.root, 'test_data')
        self.files_list = list_files(join_path(self.root, 'images'))
        self.size = len(self.files_list)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        im = Image.open(join_path(self.root, 'images', self.files_list[index]))
        template = Image.open(self.template_path)

        if self.image_shape is not None:
            im = thumbnail_image(im, self.image_shape)

        if self.transform is not None:
            im = self.transform(im)
            template = self.transform(template)

        return im, template
