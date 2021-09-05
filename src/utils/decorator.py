import argparse
from typing import Callable

from datasets import dataset_dict


def counting_script(func: Callable) -> Callable:
    def decorator(*args, **kwargs):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "-dp", "--data-path", type=str, default="../data/ILSVRC/ILSVRC2015", help="Dataset root path"
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            default="ILSVRC",
            choices=dataset_dict.keys(),
            help="Dataset name",
        )

        parser.add_argument("-i", "--image-shape", type=int, default=255, help="Image size")

        # Training parameters
        parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
        parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size")
        parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate")
        parser.add_argument("--cpu", action="store_true", help="Train using cpu only")

        return func(parser=parser, *args, **kwargs)

    return decorator
