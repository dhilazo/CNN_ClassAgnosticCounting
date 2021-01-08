import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from models.cnn_vae import ConvVAE

if __name__ == "__main__":
    image_shape = (255, 255)
    dataset_root = './data/CIFAR10'

    transform = transforms.Compose([transforms.Resize((96, 96), interpolation=Image.NEAREST),
                                    transforms.ToTensor()])

    test_set = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    model_r = ConvVAE(channels=1)
    model_g = ConvVAE(channels=1)
    model_b = ConvVAE(channels=1)

    model_r.load_state_dict(torch.load("./trained_models/ConvVAE_r.pt"))
    model_g.load_state_dict(torch.load("./trained_models/ConvVAE_g.pt"))
    model_b.load_state_dict(torch.load("./trained_models/ConvVAE_b.pt"))

    model_r.eval()
    model_g.eval()
    model_b.eval()

    index = random.randint(0, len(test_set))
    print(index)
    image, class_index = test_set[4942]

    image = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    decoded_r, _, _ = model_r(image[:, 0, :, :].unsqueeze_(0))
    decoded_g, _, _ = model_g(image[:, 1, :, :].unsqueeze_(0))
    decoded_b, _, _ = model_b(image[:, 2, :, :].unsqueeze_(0))
    decoded = torch.cat((decoded_r, decoded_g, decoded_b), 1)

    im1 = transforms.ToPILImage()(image[0]).convert("RGB")
    im2 = transforms.ToPILImage()(decoded[0]).convert("RGB")
    im1.save('../../color.jpg')
    im2.save('../../color_decoded.jpg')
    Image.fromarray(np.hstack((np.array(im1), np.array(im2)))).show()
