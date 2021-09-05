from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.system import create_dirs, join_path


class Trainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Union[torch.nn.Module, Callable],
        optimizer: Optimizer,
        run_name: str,
        device: torch.device = torch.device("cpu"),
        init_epoch: int = 0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.run_name = run_name
        self.device = device
        self.init_epoch = init_epoch

        logs_path = create_dirs(f"logs/{run_name}")
        self.train_writer = SummaryWriter(join_path(logs_path, "train"))
        self.val_writer = SummaryWriter(join_path(logs_path, "val"))

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, batch_report: int = 2000):
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.init_epoch, epochs):
            train_loss = self.train_batch_loop(epoch, train_loader, batch_report)

            self.model.eval()
            with torch.no_grad():
                val_loss = self.quick_validate(val_loader)
            self.model.train()

            torch.save(self.model.state_dict(), f"./trained_models/{self.run_name}_batch.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": val_loss,
                },
                f"./trained_models/checkpoints/{self.run_name}_checkpoint.pth",
            )

            train_mean_loss = np.mean(train_loss)
            val_mean_loss = np.mean(val_loss)

            print(f"Train loss: {train_mean_loss} Val loss: {val_mean_loss}", flush=True)

            self.train_writer.add_scalar("loss", train_mean_loss, epoch + 1)
            self.val_writer.add_scalar("loss", val_mean_loss, epoch + 1)

        print("Finished Training", flush=True)

    @abstractmethod
    def train_batch_loop(self, epoch: int, train_loader: DataLoader, batch_report: int):
        raise NotImplementedError

    @abstractmethod
    def quick_validate(self, val_loader: DataLoader) -> Sequence:
        raise NotImplementedError
