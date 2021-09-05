import argparse
from datetime import datetime
from typing import Optional

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from datasets import SpatialDensityCountingDataset, dataset_dict
from datasets.ilsvrc_dataset import ILSVRC
from models import vae_model_dict
from trainers.vae_trainer import VAETrainer
from utils import system
from utils.decorator import counting_script
from utils.mse_kld_loss import MSEKLDLoss
from utils.system import file_exists


@counting_script
def train_vae(parser: Optional[argparse.ArgumentParser] = None):
    parser = argparse.ArgumentParser(
        description="Trains a Variational Autoencoder with the specified parameters.", parents=[parser]
    )
    parser.add_argument(
        "-r",
        "--run-name",
        type=str,
        default=datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
        help="Name that will be used to store both the logs and the trained weights of the model",
    )
    parser.add_argument(
        "-v",
        "--vae-model",
        type=str,
        default="ConvVAE",
        choices=vae_model_dict.keys(),
        help="VAE model to train",
    )
    args = parser.parse_args()

    network_model = vae_model_dict[args.vae_model]
    dataset = dataset_dict[args.dataset]
    image_shape = (args.image_shape, args.image_shape)
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    print("Creating DataLoaders.", flush=True)
    if issubclass(dataset, SpatialDensityCountingDataset):
        kwargs = dict(root=args.data_path, image_shape=image_shape, transform=transform)
        if dataset == ILSVRC:
            kwargs["data_percentage"] = 0.5
    else:
        kwargs = dict(root=args.data_path, transform=transform)

    train_set = dataset(train=True, **kwargs)
    val_set = dataset(train=False, **kwargs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = network_model()
    model = model.to(device)

    criterion = MSEKLDLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    init_epoch = 0
    if file_exists("./trained_models/checkpoints/" + args.run_name + "_checkpoint.pth"):
        print("Loading checkpoint.", flush=True)
        checkpoint = torch.load(
            "./trained_models/checkpoints/" + args.run_name + "_checkpoint.pth", map_location=device
        )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            model = DataParallel(model)
            model.load_state_dict(checkpoint["model_state_dict"])
            if device == torch.device("cpu"):
                model = model.module

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"]
        print("Init epoch:", init_epoch, flush=True)

        model.train()

    system.create_dirs("trained_models")
    system.create_dirs("trained_models/checkpoints")

    trainer = VAETrainer(model, criterion, optimizer, args.run_name, device=device, init_epoch=init_epoch)
    trainer.train(args.epochs, train_loader, val_loader)

    torch.save(model.state_dict(), "./trained_models/" + args.run_name + ".pt")


if __name__ == "__main__":
    train_vae()
