import torch
from torch import Tensor, nn


class MSEKLDLoss(nn.Module):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`, and adds it to the KLD Loss
    of a Log-normal distribution.
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x: Tensor, y: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
        mse = self.mse_loss(x, y)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + kld
