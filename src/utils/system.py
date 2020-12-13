import os


def create_dirs(dir_name):
    path = os.path.join('.', dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def join_path(path, *paths):
    return os.path.join(path, *paths)


def list_files(path):
    return os.listdir(path)


def file_exists(path):
    return os.path.isfile(path)
