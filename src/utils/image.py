from PIL import Image


def create_image_grid(images, image_grid_shape):
    images_shape = images[0].size
    grid_image = Image.new('RGB', (images_shape[-2] * image_grid_shape[0], images_shape[-1] * image_grid_shape[1]))

    images_index = 0
    for row in range(0, images_shape[-2] * image_grid_shape[0], images_shape[-2]):
        for col in range(0, images_shape[-1] * image_grid_shape[1], images_shape[-1]):
            grid_image.paste(images[images_index], (row, col))
            images_index += 1
    return grid_image


def resize_image(image, final_shape):
    image.thumbnail(final_shape, Image.ANTIALIAS)
    return image


def pad_image(image, final_shape):
    image_origin = (final_shape[0] // 2 - image.size[-2] // 2, final_shape[1] // 2 - image.size[-2] // 2)
    padded_image = Image.new('RGB', final_shape)
    padded_image.paste(image, image_origin)

    return padded_image


def open_image(path):
    return Image.open(path)
