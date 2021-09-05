import argparse

import numpy as np
from tqdm import tqdm

from datasets.cifar10_count_dataset import CIFAR10CountDataset
from utils.system import create_dirs, join_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves preprocessed CIFAR10 Count dataset to speed up training stage")
    parser.add_argument("-p", "--path", type=str, default="../data/CIFAR10Count", help="Dataset root path")
    args = parser.parse_args()

    image_grid_distribution = (3, 3)
    create_dirs(args.path)

    train_set = CIFAR10CountDataset("./data/CIFAR10", image_grid_distribution, template_view="raw", train=True)
    test_set = CIFAR10CountDataset("./data/CIFAR10", image_grid_distribution, template_view="raw", train=False)

    # Save templates
    create_dirs(join_path(args.path, "templates"))
    _, templates, _ = train_set[0]
    for i, template in tqdm(enumerate(templates), desc="Extracting templates"):
        template.save(join_path(args.path, "templates", f"{train_set.class_names[i]}.jpg"), "JPEG")

    # Save train data
    create_dirs(join_path(args.path, "train"))
    create_dirs(join_path(args.path, "train", "images"))
    create_dirs(join_path(args.path, "train", "counts"))
    with open(join_path(args.path, "train", "counts", "counts.txt"), "w") as counts_file:
        for i, data in tqdm(enumerate(train_set), desc="Extracting train data"):
            image_grid, _, counts = data
            counts = [count[0] for count in counts.astype(np.int32)]
            image_grid.save(join_path(args.path, "train", "images", f"{i}.jpg"), "JPEG")
            counts_file.write(f"{' '.join(map(str, counts))}\n")

    # Save test data
    create_dirs(join_path(args.path, "test"))
    create_dirs(join_path(args.path, "test", "images"))
    create_dirs(join_path(args.path, "test", "counts"))
    with open(join_path(args.path, "test", "counts", "counts.txt"), "w") as counts_file:
        for i, data in tqdm(enumerate(train_set), desc="Extracting test data"):
            image_grid, _, counts = data
            counts = [count[0] for count in counts.astype(np.int32)]
            image_grid.save(join_path(args.path, "test", "images", f"{i}.jpg"), "JPEG")
            counts_file.write(f"{' '.join(map(str, counts))}\n")
