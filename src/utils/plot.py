import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def imshow(img: Tensor, normalized: bool = False) -> None:
    if normalized:
        # Unnormalize
        img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_sample(image_grid: Tensor, template: Tensor, count: int, normalized: bool = False) -> None:
    if normalized:
        # Unnormalize
        image_grid = image_grid / 2 + 0.5
        template = template / 2 + 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    ax1.imshow(np.rollaxis(image_grid.numpy(), 0, 3))
    ax1.set_title("Sample")

    ax2.imshow(np.rollaxis(template.numpy(), 0, 3))
    ax2.set_title("Template")

    plt.suptitle(f"Groud truth count:{count}")
    plt.show()
