import os
from typing import Sequence


def create_dirs(dir_name: str) -> str:
    path = os.path.join(".", dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def join_path(path: str, *paths) -> str:
    return os.path.join(path, *paths)


def list_files(path: str) -> Sequence[str]:
    return os.listdir(path)


def file_exists(path: str) -> bool:
    return os.path.isfile(path)
