from .density_counting import (AdaptedGenericMatchingNetwork, GenericMatchingNetwork, GMNETCNet, GmnETSCNN,
                               SiameseGenericMatchingNetwork)
from .direct_counting import ETSCNN, BasicNet, DoubleInputNet, ETCNet, ResNet, SiameseNet, SiameseResNet
from .variational_autoencoders import CNNDecoder, CNNEncoder, ConvVAE, ConvVAEGMN

counting_model_dict = {
    # Direct counting models
    "ETSCNN": ETSCNN,
    "BasicNet": BasicNet,
    "DoubleInputNet": DoubleInputNet,
    "ETCNet": ETCNet,
    "ResNet": ResNet,
    "SiameseNet": SiameseNet,
    "SiameseResNet": SiameseResNet,
    # Density couting models
    "AdaptedGMN": AdaptedGenericMatchingNetwork,
    "GMN": GenericMatchingNetwork,
    "GMNETCNet": GMNETCNet,
    "GmnETSCNN": GmnETSCNN,
    "SiameseGMN": SiameseGenericMatchingNetwork,
}

vae_model_dict = {"ConvVAE": ConvVAE, "ConvVAEGMN": ConvVAEGMN}
