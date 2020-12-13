import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from datasets.ilsvrc_dataset import ILSVRC
from models.cnn_vae import ConvVAE
from models.cnn_vae_gmn import ConvVAEGMN

if __name__ == "__main__":
    image_shape = (255, 255)
    data_root = './data/ILSVRC/ILSVRC2015'

    transform = transforms.Compose([transforms.ToTensor()])

    test_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=False, transform=transform)

    model = ConvVAEGMN()
    model.load_state_dict(torch.load("./trained_models/ConvVAE_ILSVRC_batch.pt"))
    model.eval()

    images, templates, ground_truth, count = test_set[1]
    templates = torch.reshape(templates, (1, templates.shape[-3], templates.shape[-2], templates.shape[-1]))
    decoded, _, _ = model(templates)

    im1 = transforms.ToPILImage()(templates[0]).convert("RGB")
    im2 = transforms.ToPILImage()(decoded[0]).convert("RGB")
    Image.fromarray(np.hstack((np.array(im1), np.array(im2)))).show()

    mu, logvar = model.encoder(templates)
    z = model.reparametrize(mu, logvar)
    encoded = z[0].detach().numpy()
    #encoded = torch.reshape(z, (8, 8, z.shape[-2], z.shape[-1]))

    # Plot
    fig, axs = plt.subplots(8, 8, figsize=(10, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.0005)

    axs = axs.ravel()
    plt.axis('off')
    for i in range(64):
        # axs[i].contourf(encoded[i], 5, cmap=plt.cm.Oranges)
        axs[i].imshow(encoded[i], cmap=plt.cm.Greys)
        axs[i].set_axis_off()
    plt.suptitle('Encoded features')
    plt.show()
