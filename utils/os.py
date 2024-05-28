import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)

def list_visible(path, subset='a'):
    """
    Lists all visible (non-hidden) files and subdirectories in a directory.
    :param path: Path to directory.
    :param subset: 'a' (as in 'all'), 'f' (as in 'files'), or 'd' (as in 'directories') to specify which subset of
    visible contents to list.
    :return: List of files and subdirectories.
    """
    contents = os.listdir(path)
    contents = [item for item in contents if not item.startswith('.')]
    if subset == 'a':
        return contents
    elif subset == 'f':
        return [item for item in contents if os.path.isfile(os.path.join(path, item))]
    elif subset == 'd':
        return [item for item in contents if os.path.isdir(os.path.join(path, item))]
    else:
        raise ValueError('Unsupported value for argument "subset". Use "a" for all, "f" for files, or "d" for '
                         'directories.')


def load_config(path):
    """
    Loads a JSON configuration file into a dictionary.
    :param path: Path to `.json` file.
    :return: Configuration dictionary.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def remove(path):
    """
    Removes a file or directory.
    :param path: Path to item.
    :return: None.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    else:
        raise RuntimeError('Target is neither a file nor a directory.')

def reset(path, to_create):
    """
    Resets a file or directory to become empty. This means that if the path's item already exists, it is deleted and
    then re-created to be empty. If the path's item does not exist, it is simply created to be empty.
    :param path: Path to item.
    :param to_create: 'f' (as in 'file'), or 'd' (as in 'directory') to specify which type of item to create.
    :return: None.
    """
    if os.path.exists(path):
        remove(path)
    if to_create == 'f':
        open(path, 'w').close()
    elif to_create == 'd':
        os.mkdir(path)
    else:
        raise ValueError('Unsupported value for argument "to_create". Use "f" for file or "d" for directory.')