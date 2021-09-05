from .cnn_decoder import CNNDecoder
from .cnn_encoder import CNNEncoder
from .conv_vae import ConvVAE
from .conv_vae_gmn import ConvVAEGMN

variational_autoencoders = [CNNDecoder, CNNEncoder, ConvVAE, ConvVAEGMN]
