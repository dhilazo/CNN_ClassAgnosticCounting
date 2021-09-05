from typing import Sequence

from torch.utils.data import DataLoader

from .trainer import Trainer


class DensityCountTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DensityCountTrainer, self).__init__(*args, **kwargs)

    def train_batch_loop(self, epoch: int, train_loader: DataLoader, batch_report: int):
        running_loss = 0.0
        train_loss = []

        loss_count = 0
        for data in train_loader:
            images, templates, ground_truth, count, resized_template = data

            images = images.to(self.device)
            templates = templates.to(self.device)
            ground_truth = ground_truth.to(self.device)
            resized_template = resized_template.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images, templates, resized_template)

            loss = self.criterion(outputs, ground_truth)
            loss.backward()
            self.optimizer.step()
            loss_count += 1

            # Print statistics
            train_loss.append(loss.item())
            running_loss += loss.item()
            if loss_count % batch_report == batch_report - 1:  # print every batch_report mini-batches
                print(
                    f"[{epoch + 1}, {loss_count + 1}/{len(train_loader)}] loss: {running_loss / batch_report}",
                    flush=True,
                )
                running_loss = 0.0

    def quick_validate(self, val_loader: DataLoader) -> Sequence:
        epoch_val_loss = []
        for data in val_loader:
            images, templates, ground_truth, count, resized_template = data

            images = images.to(self.device)
            templates = templates.to(self.device)
            ground_truth = ground_truth.to(self.device)
            resized_template = resized_template.to(self.device)

            outputs = self.model(images, templates, resized_template)

            loss = self.criterion(outputs, ground_truth)
            epoch_val_loss.append(loss.item())

        return epoch_val_loss
