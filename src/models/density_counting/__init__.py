from .adapted_gmn import AdaptedGenericMatchingNetwork
from .gmn import GenericMatchingNetwork
from .gmn_etcnet import GMNETCNet
from .gmn_etscnn import GmnETSCNN
from .siamese_gmn import SiameseGenericMatchingNetwork

density_counting_models = [
    AdaptedGenericMatchingNetwork,
    GenericMatchingNetwork,
    GMNETCNet,
    GmnETSCNN,
    SiameseGenericMatchingNetwork,
]
