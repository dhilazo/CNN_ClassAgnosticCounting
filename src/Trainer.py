import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.system import create_log_dirs, join_path


class Trainer:
    def __init__(self, model, criterion, optimizer, run_name):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        logs_path = create_log_dirs(run_name)
        self.train_writer = SummaryWriter(join_path(logs_path, 'train'))
        self.val_writer = SummaryWriter(join_path(logs_path, 'val'))

    def train(self, epochs, train_loader, val_loader, batch_report=2000):
        for epoch in range(epochs):  # loop over the dataset multiple times
            since_epoch = time.time()
            running_loss = 0.0
            train_loss = []

            loss_count = 0
            for i, data in enumerate(train_loader, 0):
                since_batch = time.time()
                # get the inputs; data is a list of [inputs, labels]
                image_grids, templates, counts = data

                templates = np.asarray(templates, dtype=object)
                for i in range(len(templates)):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image_grids, templates[i])  # TODO fix templates order

                    loss = self.criterion(outputs, counts[:, i])
                    loss.backward()
                    self.optimizer.step()
                    loss_count += 1

                    # Print statistics
                    train_loss.append(loss.item())
                    running_loss += loss.item()
                    if loss_count % batch_report == batch_report - 1:  # print every batch_report mini-batches
                        print(
                            f'[{epoch + 1}, {loss_count + 1}/{len(train_loader)*len(templates)}] '
                            f'loss: {running_loss / batch_report}\tTime: {round(time.time() - since_batch, 2)}s')
                        running_loss = 0.0
                        since_batch = time.time()

            val_loss = self.quick_validate(val_loader)

            train_mean_loss = np.mean(train_loss)
            val_mean_loss = np.mean(val_loss)
            print(f'Losses: {train_mean_loss} {val_mean_loss}')
            print(f'Epoch time: {round(time.time()-since_epoch, 2)}s')

            self.train_writer.add_scalar('loss', train_mean_loss, epoch + 1)
            self.val_writer.add_scalar('loss', val_mean_loss, epoch + 1)

        print('Finished Training')

    def quick_validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            for ii, (image_grids, templates, counts) in enumerate(val_loader):
                templates = np.asarray(templates, dtype=object)
                for i in range(len(templates)):
                    prediction = self.model(image_grids, templates[i])

                    loss = self.criterion(prediction, counts[:, i])

                    epoch_val_loss.append(loss.item())
        self.model.train()
        return epoch_val_loss
