from PIL import Image
from torchvision import datasets

from utils.image import thumbnail_image
from utils.system import join_path, list_files


class SavedCIFAR10CountDataset(datasets.CIFAR10):
    """
    The CIFAR10CountDataset dataset contains grid images of multiple CIFAR10 examples, as well as the counts
    for each class inside the image grid.
    A sample from this dataset will return the image grid, a template for each of the classes contained in the image,
    their appearance count and their class labels.

    Args:
        - root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
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

    def __init__(self, root,  train=True, transform=None):
        self.root = root
        self.class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.templates_path = join_path(self.root, 'templates')
        self.transform = transform
        if train:
            self.root = join_path(self.root, 'train')
        else:
            self.root = join_path(self.root, 'test')
        self.files_list = list_files(join_path(self.root, 'images'))
        self.size = len(self.files_list)

        with open(join_path(self.root, 'counts', 'counts.txt'), 'r') as counts_file:
            self.count_lines = counts_file.readlines()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        im = Image.open(join_path(self.root, 'images', self.files_list[index]))
        templates = self.open_templates()
        counts = self.count_lines[index].split(' ')
        counts = [int(string) for string in counts]

        if self.transform is not None:
            im = self.transform(im)

        return im, templates, counts

    def open_templates(self):
        templates = []
        for class_name in self.class_names:
            template_name = class_name + '.jpg'
            template = Image.open(join_path(self.templates_path, template_name))
            if self.transform is not None:
                template = self.transform(template)
            templates.append(template)
        return templates
