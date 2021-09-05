from typing import Sequence

from torch.utils.data import DataLoader

from trainers.trainer import Trainer


class VAETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)

    def train_batch_loop(self, epoch: int, train_loader: DataLoader, batch_report: int):
        running_loss = 0.0
        train_loss = []

        loss_count = 0
        for i, data in enumerate(train_loader, 0):
            _, templates, _, _ = data

            templates = templates.to(self.device)

            self.optimizer.zero_grad()

            decoded, mu, logvar = self.model(templates)

            loss = self.criterion(decoded, templates, mu, logvar)
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
            _, templates, _, _ = data
            templates = templates.to(self.device)

            decoded, mu, logvar = self.model(templates)

            loss = self.criterion(decoded, templates, mu, logvar)
            epoch_val_loss.append(loss.item())

        return epoch_val_loss
