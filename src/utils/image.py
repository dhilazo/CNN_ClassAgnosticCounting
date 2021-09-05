from typing import Sequence, Tuple

from PIL import Image


def create_image_grid(images: Sequence, image_grid_shape: Tuple[int, int]) -> Image.Image:
    """Creates an image grid with the given images and with the specified shape"""
    images_shape = images[0].size
    grid_image = Image.new("RGB", (images_shape[-2] * image_grid_shape[0], images_shape[-1] * image_grid_shape[1]))

    images_index = 0
    for row in range(0, images_shape[-2] * image_grid_shape[0], images_shape[-2]):
        for col in range(0, images_shape[-1] * image_grid_shape[1], images_shape[-1]):
            grid_image.paste(images[images_index], (row, col))
            images_index += 1
    return grid_image


def resize_image(image: Image.Image, final_shape: Tuple[int, int]) -> Image.Image:
    """Resizes given Image to specified shape"""
    resized_image = image.resize(final_shape)
    return resized_image


def pad_image(image: Image.Image, final_shape: Tuple[int, int]) -> Image.Image:
    """Adds padding to given Image"""
    image_origin = (final_shape[0] // 2 - image.size[-2] // 2, final_shape[1] // 2 - image.size[-1] // 2)
    padded_image = Image.new("RGB", final_shape)
    padded_image.paste(image, image_origin)

    return padded_image


def repeat_image(image: Image.Image, final_shape: Tuple[int, int]) -> Image.Image:
    """Returns an image mosaic of the given image and the specified shape"""
    final_image = Image.new("RGB", final_shape)
    for width in range(0, final_shape[0], image.size[-2]):
        for height in range(0, final_shape[1], image.size[-1]):
            final_image.paste(image, (width, height))
    return final_image


def open_image(path: str) -> Image.Image:
    """Returns Image object in path"""
    return Image.open(path)


def thumbnail_image(image: Image.Image, final_shape: Tuple[int, int]) -> Image.Image:
    """Reduces the size of the given Image to the specified shape"""
    image.thumbnail(final_shape, Image.ANTIALIAS)
    return image


def add_square_padding(image: Image.Image, image_shape: Tuple[int, int], output_mode: str = "RGB") -> Image.Image:
    """Transforms the given Image into a 1:1 resolution by adding padding to the smaller axis"""
    padding = Image.new("RGBA", image_shape)
    x = int(padding.size[0] / 2 - image.size[0] / 2)
    y = int(padding.size[1] / 2 - image.size[1] / 2)
    image = image.convert("RGBA")
    image = paste_image(padding, image, (x, y)).convert(output_mode)
    return image


def paste_image(background: Image.Image, foreground: Image.Image, center: Tuple[int, int]) -> Image.Image:
    """Pastes an Image on top of the other in a given position"""
    background.paste(foreground, center, foreground)
    return background
