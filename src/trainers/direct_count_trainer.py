from typing import Sequence

from torch.utils.data import DataLoader

from .trainer import Trainer


class DirectCountTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DirectCountTrainer, self).__init__(*args, **kwargs)

    def train_batch_loop(self, epoch: int, train_loader: DataLoader, batch_report: int):
        running_loss = 0.0
        train_loss = []

        loss_count = 0
        for data in train_loader:
            image_grids, templates, counts = data

            image_grids = image_grids.to(self.device)
            counts = counts.to(self.device)

            for template_idx in range(len(templates)):
                self.optimizer.zero_grad()

                current_template = templates[template_idx]
                current_template = current_template.to(self.device)

                outputs = self.model(image_grids, current_template)

                loss = self.criterion(outputs, counts[:, template_idx])
                loss.backward()
                self.optimizer.step()
                loss_count += 1

                # Print statistics
                train_loss.append(loss.item())
                running_loss += loss.item()
                if loss_count % batch_report == batch_report - 1:  # print every batch_report mini-batches
                    print(
                        f"[{epoch + 1}, {loss_count + 1}/{len(train_loader) * len(templates)}] "
                        f"loss: {running_loss / batch_report}",
                        flush=True,
                    )
                    running_loss = 0.0

    def quick_validate(self, val_loader: DataLoader) -> Sequence:
        epoch_val_loss = []
        for image_grids, templates, counts in val_loader:
            image_grids = image_grids.to(self.device)
            counts = counts.to(self.device)

            for template_idx in range(len(templates)):
                current_template = templates[template_idx]
                current_template = current_template.to(self.device)
                prediction = self.model(image_grids, current_template)

                loss = self.criterion(prediction, counts[:, template_idx])
                epoch_val_loss.append(loss.item())

        return epoch_val_loss
