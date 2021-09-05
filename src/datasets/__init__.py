from .carpk_dataset import CARPK
from .cifar10_count_dataset import CIFAR10CountDataset
from .ilsvrc_dataset import ILSVRC
from .preprocessed_cifar10_count_dataset import PreprocessedCIFAR10CountDataset
from .spatial_density_counting_dataset import SpatialDensityCountingDataset

dataset_dict = {
    "CARPK": CARPK,
    "CIFAR10": CIFAR10CountDataset,
    "ILSVRC": ILSVRC,
    "PreprocessedCIFAR10": PreprocessedCIFAR10CountDataset,
}
