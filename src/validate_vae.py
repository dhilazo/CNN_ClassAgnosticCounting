import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from models.cnn_vae import ConvVAE

if __name__ == "__main__":
    dataset_root = './data/CIFAR10'

    transform = transforms.Compose([transforms.Resize((96, 96), interpolation=Image.NEAREST),
                                    transforms.ToTensor()])

    test_set = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)

    model = ConvVAE()
    model.load_state_dict(torch.load("./trained_models/ConvVAE.pt"))

    image, _ = test_set[1]
    image = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    decoded, _, _ = model(image)

    im1 = transforms.ToPILImage()(image[0]).convert("RGB")
    im2 = transforms.ToPILImage()(decoded[0]).convert("RGB")
    Image.fromarray(np.hstack((np.array(im1), np.array(im2)))).show()
