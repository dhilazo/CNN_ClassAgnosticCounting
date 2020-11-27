import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from datasets.ilsvrc_dataset import ILSVRC
from models.etcnet_model import ETCNet
from models.etscnn_model import ETSCNN
from models.gmn import GenericMatchingNetwork
from train_gmn import Trainer_GMN
from utils import system


def save_template(train_loader, classes):
    found = []
    # get some random training images
    dataiter = iter(train_loader)
    while len(found) != len(classes):
        images, labels = dataiter.next()
        for img, label in zip(images, labels):
            if label not in found:
                found.append(label)
                # Save and remove
                torch.save(img, './data/templates/' + classes[label] + '.pt')


if __name__ == "__main__":
    run_name = 'GMN_short'
    network_model = GenericMatchingNetwork
    epochs = 100
    image_shape = (255, 255)
    batch_size = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])

    print("Creating DataLoaders.", flush=True)
    data_root = './data/ILSVRC/ILSVRC2015'
    train_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.8, train=True, transform=transform)
    val_set = ILSVRC(data_root, image_shape=image_shape, train=False, transform=transform)

    # train_len = len(train_set)
    # train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("DataLoaders created.", flush=True)

    model = network_model(output_matching_size=(255 // 4, 255 // 4))
    model = model.to(device)
    if isinstance(model, ETCNet) or isinstance(model, ETSCNN):
        model.load_vae('./trained_models/ConvVAE.pt')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    system.create_dirs('trained_models')
    trainer = Trainer_GMN(model, criterion, optimizer, run_name, device=device)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')

    # trainer.evaluate(test_loader)
