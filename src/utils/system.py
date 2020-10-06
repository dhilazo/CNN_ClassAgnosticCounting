import os


def create_log_dirs(run_name):
    path = os.path.join('.', 'logs', run_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def join_path(path, *paths):
    return os.path.join(path, *paths)
