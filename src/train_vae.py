import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from models.cnn_vae import ConvVAE
from utils import system
from utils.system import join_path, create_dirs


class VAETrainer:
    def __init__(self, model, criterion, optimizer, run_name, device=torch.device('cpu')):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.run_name = run_name

        logs_path = create_dirs(f'logs/{run_name}')
        self.train_writer = SummaryWriter(join_path(logs_path, 'train'))
        self.val_writer = SummaryWriter(join_path(logs_path, 'val'))

    def train(self, epochs, train_loader, val_loader, batch_report=2000):
        for epoch in range(epochs):  # loop over the dataset multiple times
            since_epoch = time.time()
            running_loss = 0.0
            train_loss = []

            loss_count = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                decoded, mu, logvar = self.model(images)

                loss = self.criterion(decoded, images)
                loss.backward()
                self.optimizer.step()
                loss_count += 1

                # Print statistics
                train_loss.append(loss.item())
                running_loss += loss.item()
                if loss_count % batch_report == batch_report - 1:  # print every batch_report mini-batches
                    print(
                        f'[{epoch + 1}, {loss_count + 1}/{len(train_loader)}] '
                        f'loss: {running_loss / batch_report}', flush=True)
                    running_loss = 0.0

            val_loss = self.quick_validate(val_loader)  # TODO save model
            torch.save(self.model.state_dict(), './trained_models/' + self.run_name + '_batch.pt')

            train_mean_loss = np.mean(train_loss)
            val_mean_loss = np.mean(val_loss)

            print(f'Train loss: {train_mean_loss} Val loss: {val_mean_loss}', flush=True)
            print(f'Epoch time: {round(time.time() - since_epoch, 2)}s', flush=True)

            self.train_writer.add_scalar('loss', train_mean_loss, epoch + 1)
            self.val_writer.add_scalar('loss', val_mean_loss, epoch + 1)

        print('Finished Training', flush=True)

    def quick_validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            for ii, data in enumerate(val_loader):
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                decoded, mu, logvar = self.model(images)

                loss = self.criterion(decoded, images)

                epoch_val_loss.append(loss.item())
        self.model.train()
        return epoch_val_loss

    def evaluate(self, test_loader):
        self.model.eval()

        max_test_loss = 0
        test_accumulated_loss = 0
        total_batches = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                decoded, mu, logvar = self.model(images)

                loss = self.criterion(decoded, images)

                if loss > max_test_loss:
                    max_test_loss = loss
                test_accumulated_loss += loss
                total_batches += 1

        print(f'Average loss: {test_accumulated_loss / total_batches}')
        print(f'Max loss: {max_test_loss}')


if __name__ == "__main__":
    run_name = 'ConvVAE'
    epochs = 100
    batch_size = 16
    dataset_root = './data/CIFAR10'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.Resize((96, 96), interpolation=Image.NEAREST),
                                    transforms.ToTensor()])

    train_set = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)

    train_len = len(train_set)
    train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConvVAE()

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    system.create_dirs('trained_models')
    trainer = VAETrainer(model, criterion, optimizer, run_name, device=device)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')

    trainer.evaluate(test_loader)
