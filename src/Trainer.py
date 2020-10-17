import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.system import join_path, create_dirs


class Trainer:
    def __init__(self, model, criterion, optimizer, run_name, device='cpu'):
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
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            since_epoch = time.time()
            running_loss = 0.0
            train_loss = []

            loss_count = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                image_grids, templates, counts = data

                image_grids = image_grids.to(self.device)
                counts = counts.to(self.device)

                for i in range(len(templates)):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    current_template = templates[i]
                    current_template = current_template.to(self.device)

                    # forward + backward + optimize
                    outputs = self.model(image_grids, current_template)  # TODO fix templates order

                    loss = self.criterion(outputs, counts[:, i])
                    loss.backward()
                    self.optimizer.step()
                    loss_count += 1

                    # Print statistics
                    train_loss.append(loss.item())
                    running_loss += loss.item()
                    if loss_count % batch_report == batch_report - 1:  # print every batch_report mini-batches
                        print(
                            f'[{epoch + 1}, {loss_count + 1}/{len(train_loader) * len(templates)}] '
                            f'loss: {running_loss / batch_report}')
                        running_loss = 0.0

            val_loss = self.quick_validate(val_loader)  # TODO save model
            torch.save(self.model.state_dict(), './trained_models/' + self.run_name + '_batch.pt')

            train_mean_loss = np.mean(train_loss)
            val_mean_loss = np.mean(val_loss)

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            print(f'Train loss: {train_mean_loss} Val loss: {val_mean_loss}')
            print(f'Epoch time: {round(time.time() - since_epoch, 2)}s')

            self.train_writer.add_scalar('loss', train_mean_loss, epoch + 1)
            self.val_writer.add_scalar('loss', val_mean_loss, epoch + 1)

        print('Finished Training')

    def quick_validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            for ii, (image_grids, templates, counts) in enumerate(val_loader):
                image_grids = image_grids.to(self.device)
                counts = counts.to(self.device)

                for i in range(len(templates)):
                    current_template = templates[i]
                    current_template = current_template.to(self.device)
                    prediction = self.model(image_grids, current_template)

                    loss = self.criterion(prediction, counts[:, i])

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
                image_grids, templates, counts = data

                image_grids = image_grids.to(self.device)
                counts = counts.to(self.device)

                for i in range(len(templates)):
                    current_template = templates[i]
                    current_template = current_template.to(self.device)
                    outputs = self.model(image_grids, current_template)

                    loss = self.criterion(outputs, counts[:, i])

                    if loss > max_test_loss:
                        max_test_loss = loss
                    test_accumulated_loss += loss
                    total_batches += 1

        print(f'Average loss: {test_accumulated_loss / total_batches}')
        print(f'Max loss: {max_test_loss}')
