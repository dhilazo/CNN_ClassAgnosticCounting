import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.ilsvrc_dataset import ILSVRC
from models.cnn_vae import ConvVAE
from models.cnn_vae_gmn import ConvVAEGMN
from utils import system
from utils.klmse import MSEKLDLoss
from utils.system import join_path, create_dirs, file_exists


class VAETrainer:
    def __init__(self, model, criterion, optimizer, run_name, device=torch.device('cpu'), init_epoch=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.run_name = run_name

        logs_path = create_dirs(f'logs/{run_name}')
        self.train_writer = SummaryWriter(join_path(logs_path, 'train'))
        self.val_writer = SummaryWriter(join_path(logs_path, 'val'))

        self.init_epoch = init_epoch

    def train(self, epochs, train_loader, val_loader, batch_report=2000):
        for epoch in range(init_epoch + 1, epochs):  # loop over the dataset multiple times
            since_epoch = time.time()
            running_loss = 0.0
            train_loss = []

            loss_count = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, templates, ground_truth, count = data

                templates = templates.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                decoded, mu, logvar = self.model(templates)

                loss = MSEKLDLoss()(decoded, templates, mu, logvar)
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

            val_loss = self.quick_validate(val_loader)
            torch.save(self.model.state_dict(), './trained_models/' + self.run_name + '_batch.pt')

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
            }, './trained_models/checkpoints/' + self.run_name + '_checkpoint.pth')

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
                images, templates, ground_truth, count = data

                templates = templates.to(self.device)

                # forward + backward + optimize
                decoded, mu, logvar = self.model(templates)

                loss = MSEKLDLoss()(decoded, templates, mu, logvar)

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
                images, templates, ground_truth, count = data

                templates = templates.to(self.device)

                # forward + backward + optimize
                decoded, mu, logvar = self.model(templates)

                loss = MSEKLDLoss()(decoded, templates, mu, logvar)

                if loss > max_test_loss:
                    max_test_loss = loss
                test_accumulated_loss += loss
                total_batches += 1

        print(f'Average loss: {test_accumulated_loss / total_batches}')
        print(f'Max loss: {max_test_loss}')


if __name__ == "__main__":
    run_name = 'ConvVAE_firstConv_GMN'
    epochs = 100
    batch_size = 64
    image_shape = (255, 255)
    data_root = './data/ILSVRC/ILSVRC2015'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=True, transform=transform)
    val_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=False, transform=transform)

    # train_len = len(train_set)
    # train_set, val_set = random_split(train_set, [int(train_len * 0.8), int(train_len * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConvVAE()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    init_epoch = 0
    # model = nn.DataParallel(model)
    if file_exists('./trained_models/checkpoints/' + run_name + '_checkpoint.pth'):
        print("Loading checkpoint.", flush=True)
        checkpoint = torch.load('./trained_models/checkpoints/' + run_name + '_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        print("Init epoch:", init_epoch, flush=True)

        model.train()

    system.create_dirs('trained_models')
    system.create_dirs('trained_models/checkpoints')
    trainer = VAETrainer(model, criterion, optimizer, run_name, device=device, init_epoch=init_epoch)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')

    # trainer.evaluate(test_loader)
