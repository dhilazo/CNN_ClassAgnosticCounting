from typing import Tuple

import torch.nn as nn
import torch.utils.data
from torch import Tensor

from models.variational_autoencoders.cnn_decoder import CNNDecoder
from models.variational_autoencoders.cnn_encoder import CNNEncoder


class ConvVAE(nn.Module):
    def __init__(self, channels: int = 3):
        super(ConvVAE, self).__init__()

        self.encoder = CNNEncoder(channels=channels)

        self.decoder = CNNDecoder(channels=channels)

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.type_as(mu)

        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
