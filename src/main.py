import argparse
from datetime import datetime
from typing import Optional

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from datasets import SpatialDensityCountingDataset, dataset_dict
from datasets.ilsvrc_dataset import ILSVRC
from models import counting_model_dict
from models.density_counting import density_counting_models
from models.density_counting.gmn_etcnet import GMNETCNet
from models.direct_counting.etcnet import ETCNet
from models.direct_counting.etscnn import ETSCNN
from trainers.density_count_trainer import DensityCountTrainer
from trainers.direct_count_trainer import DirectCountTrainer
from utils import system
from utils.decorator import counting_script
from utils.system import file_exists


@counting_script
def train_counting_model(parser: Optional[argparse.ArgumentParser] = None):
    parser = argparse.ArgumentParser(
        description="Trains a counting model with the specified parameters.", parents=[parser]
    )
    parser.add_argument(
        "-r",
        "--run-name",
        type=str,
        default=datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
        help="Name that will be used to store both the logs and the trained weights of the model",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="SiameseGMN",
        choices=counting_model_dict.keys(),
        help="Counting model",
    )
    parser.add_argument(
        "-vp", "--vae-path", type=str, default="../trained_models/ConvVAE.pt", help="Path to pretrained VAE weights"
    )
    args = parser.parse_args()

    network_model = counting_model_dict[args.model]
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
    print("DataLoaders created.", flush=True)

    if network_model in density_counting_models:
        model = network_model(output_matching_size=(image_shape[0] // 4, image_shape[1] // 4))
    else:
        model = network_model(output_size=1)
    model = model.to(device)
    if network_model in [ETCNet, ETSCNN, GMNETCNet]:
        model.load_vae(args.vae_path)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    init_epoch = 0

    if file_exists(f"./trained_models/checkpoints/{args.run_name}_checkpoint.pth"):
        print("Loading checkpoint.", flush=True)
        checkpoint = torch.load(f"./trained_models/checkpoints/{args.run_name}_checkpoint.pth", map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint["model_state_dict"])
            if device == torch.device("cpu"):
                model = model.module

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"] + 1
        print("Init epoch:", init_epoch, flush=True)

        model.train()

    system.create_dirs("trained_models")
    system.create_dirs("trained_models/checkpoints")

    trainer_class = DirectCountTrainer if network_model not in density_counting_models else DensityCountTrainer
    trainer = trainer_class(model, criterion, optimizer, args.run_name, device=device, init_epoch=init_epoch)
    trainer.train(args.epochs, train_loader, val_loader)

    torch.save(model.state_dict(), f"./trained_models/{args.run_name}.pt")


if __name__ == "__main__":
    train_counting_model()
