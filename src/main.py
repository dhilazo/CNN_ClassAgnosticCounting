import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from datasets.ilsvrc_dataset import ILSVRC
from models.etcnet_model import ETCNet
from models.etscnn_model import ETSCNN
from models.gmn_etcnet import GMNETCNet
from train_gmn import Trainer_GMN
from utils import system
from utils.system import file_exists

if __name__ == "__main__":
    run_name = 'GMNETCNet_CIFAR'
    network_model = GMNETCNet
    epochs = 100
    image_shape = (255, 255)
    batch_size = 256

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])

    print("Creating DataLoaders.", flush=True)
    data_root = './data/ILSVRC/ILSVRC2015'
    train_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=True, transform=transform)
    val_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("DataLoaders created.", flush=True)

    model = network_model(output_matching_size=(255 // 4, 255 // 4))
    model = model.to(device)
    if isinstance(model, ETCNet) or isinstance(model, ETSCNN) or isinstance(model, GMNETCNet):
        model.load_vae('./trained_models/ConvVAE.pt')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    init_epoch = 0
    model = nn.DataParallel(model)  # If the saved model is a dataparallel already, otherwise do after the if
    if file_exists('./trained_models/checkpoints/' + run_name + '_checkpoint.pth'):
        print("Loading checkpoint.", flush=True)
        checkpoint = torch.load('./trained_models/checkpoints/' + run_name + '_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        print("Init epoch:", init_epoch, flush=True)

        model.train()

    system.create_dirs('trained_models')
    system.create_dirs('trained_models/checkpoints')

    trainer = Trainer_GMN(model, criterion, optimizer, run_name, device=device, init_epoch=init_epoch)
    trainer.train(epochs, train_loader, val_loader)

    torch.save(model.state_dict(), './trained_models/' + run_name + '.pt')
