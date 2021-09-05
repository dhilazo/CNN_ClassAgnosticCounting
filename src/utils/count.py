from typing import List, Tuple

import numpy as np
from scipy import ndimage


def get_cornerless_mask(shape: Tuple[int, int]) -> np.ndarray:
    """Returns cornerless matrix of ones with given shape"""
    mask = np.ones(shape)
    mask[0, 0] = 0
    mask[-1, 0] = 0
    mask[0, -1] = 0
    mask[-1, -1] = 0

    return mask


def find_maximums(matrix: np.ndarray, footprint: np.ndarray) -> List[Tuple[int, int]]:
    data_max = ndimage.maximum_filter(matrix, footprint=footprint)
    maxima = matrix == data_max
    data_min = ndimage.minimum_filter(matrix, footprint=footprint)
    diff = (data_max - data_min) > (matrix.max() / 3)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    coordinates = [((dx.start + dx.stop - 1) // 2, (dy.start + dy.stop - 1) // 2) for dy, dx in slices]

    return coordinates


def count_local_maximums(matrix: np.ndarray) -> int:
    """Counts the local maximums in given matrix"""
    footprint_3x3 = get_cornerless_mask((3, 3))
    footprint_7x7 = get_cornerless_mask((7, 7))

    coordinates_small = find_maximums(matrix, footprint_3x3)
    coordinates_big = find_maximums(matrix, footprint_7x7)

    coordinates = set(coordinates_big + coordinates_small)

    return len(coordinates)
