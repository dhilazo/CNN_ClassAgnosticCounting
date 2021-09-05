import argparse
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import dataset_dict
from models import counting_model_dict
from models.density_counting import density_counting_models
from utils.count import count_local_maximums
from utils.decorator import counting_script


@counting_script
def test_performance(parser: Optional[argparse.ArgumentParser] = None):
    parser = argparse.ArgumentParser(
        description="Tests the performance of a trained counting model with several metrics, "
        "such as: MSE, MAE and Accuracy.",
        parents=[parser],
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
        "-wp",
        "--weights-path",
        type=str,
        default="../trained_models/SiameseGMN_batch.pt",
        help="Path to trained weights",
    )
    args = parser.parse_args()

    network_model = counting_model_dict[args.model]
    dataset = dataset_dict[args.dataset]
    image_shape = (args.image_shape, args.image_shape)
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    test_set = dataset(root=args.data_path, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if network_model in density_counting_models:
        model = network_model(output_matching_size=(image_shape[0] // 4, image_shape[1] // 4))
    else:
        model = network_model(output_size=1)

    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    except RuntimeError:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.weights_path))
        if device == torch.device("cpu"):
            model = model.module

    model = model.to(device)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        evaluate_func = (
            evaluate_direct_count if network_model not in density_counting_models else evaluate_density_count
        )
        accuracies, mae_losses, mse_losses = evaluate_func(device, mae_criterion, model, mse_criterion, test_loader)
    print("MSE Loss:", np.mean(mse_losses), flush=True)
    print("MAE Loss:", np.mean(mae_losses), flush=True)
    print("Avg Accuracy:", np.mean(accuracies) * 100, "%", flush=True)


def evaluate_density_count(device, mae_criterion, model, mse_criterion, test_loader):
    mse_losses = []
    mae_losses = []
    accuracies = []
    for batch, data in tqdm(enumerate(test_loader), leave=False, desc="Evaluating batch"):
        images, templates, _, count, resized_template = data

        images = images.to(device)
        templates = templates.to(device)
        resized_template = resized_template.to(device)

        correct = 0
        outputs = model(images, templates, resized_template)
        outputs = torch.FloatTensor(
            [count_local_maximums(outputs[i].detach().cpu().numpy()[0]) for i in range(len(outputs))]
        )
        mse_losses.append(mse_criterion(outputs, count).item())
        mae_losses.append(mae_criterion(outputs, count).item())

        correct += sum(
            np.around(np.reshape(outputs.cpu().numpy(), len(outputs)), decimals=0)
            == np.reshape(count.cpu().numpy(), len(outputs))
        )
        accuracies.append(correct / len(images))
    return accuracies, mae_losses, mse_losses


def evaluate_direct_count(device, mae_criterion, model, mse_criterion, test_loader):
    mse_losses = []
    mae_losses = []
    accuracies = []
    for batch, data in tqdm(enumerate(test_loader), leave=False, desc="Evaluating batch"):
        image_grids, templates, counts = data

        image_grids = image_grids.to(device)
        counts = counts.to(device)

        for i in range(len(templates)):
            correct = 0
            current_template = templates[i]
            current_template = current_template.to(device)
            outputs = model(image_grids, current_template)
            mse_losses.append(mse_criterion(outputs, counts[:, i]).item())
            mae_losses.append(mae_criterion(outputs, counts[:, i]).item())

            correct += sum(
                np.around(np.reshape(outputs.cpu().numpy(), len(outputs)), decimals=0)
                == np.reshape(counts[:, i].cpu().numpy(), len(outputs))
            )
            accuracies.append(correct / len(image_grids))
    return accuracies, mae_losses, mse_losses


if __name__ == "__main__":
    test_performance()
