import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from models.cnn_decoder import CNNDecoder
from models.cnn_encoder import CNNEncoder


class ConvVAE(nn.Module):
    def __init__(self, channels=3):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = CNNEncoder(channels=channels)

        # Decoder
        self.decoder = CNNDecoder(channels=channels)

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.type_as(mu)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
