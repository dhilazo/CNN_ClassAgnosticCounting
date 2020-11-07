import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split, DataLoader

from datasets.saved_cifar10_count_dataset import SavedCIFAR10CountDataset
from models.etcnet_model import ETCNet
from trainer import Trainer
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
    run_name = 'test'
    network_model = ETCNet
    epochs = 100
    image_grid_distribution = (3, 3)
    batch_size = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])

    print("Creating DataLoaders.", flush=True)
    data_root = './data/CIFAR10Count'
    train_set = SavedCIFAR10CountDataset(data_root, train=True, transform=transform)
    test_set = SavedCIFAR10CountDataset(data_root, train=False, transform=transform)

    train_len = len(train_set)
    train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("DataLoaders created.", flush=True)

    model = network_model(output_size=1)
    model = model.to(device)
    if isinstance(model, ETCNet):
        model.load_vae(
            ['./trained_models/ConvVAE_r.pt', './trained_models/ConvVAE_g.pt', './trained_models/ConvVAE_b.pt'])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    system.create_dirs('trained_models')
    trainer = Trainer(model, criterion, optimizer, run_name, device=device)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')

    trainer.evaluate(test_loader)
