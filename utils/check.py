import logging
import os

logger = logging.getLogger(__name__)


def check_args(args):
    """
    Ensures that the main program's arguments are valid. Immediately raises an error if found.
    :param args: Arguments namespace.
    :return: True if valid.
    """
    # config_path
    if not os.path.isfile(args.config_path):
        raise ValueError('Argument config_path is not a file')

    # data_path
    if not os.path.isdir(args.data_path):
        raise ValueError('Argument data_path is not a directory')

    # mode
    if args.mode not in ['train', 'predict']:
        raise ValueError('Argument mode must be either "train" or "predict"')

    # save_path
    if not os.path.isdir(args.save_path):
        raise ValueError('Argument save_path is not a directory')

    check_args_load_ckpt(args.load_ckpt)
    check_args_optimizer(args.optimizer)
    return True

def check_args_load_ckpt(load_ckpt):
    """
    Ensures that the load_ckpt argument is valid. Immediately raises an error if found.
    :param load_ckpt: The load_ckpt argument supplied as an argument of the main program.
    :return: True if valid.
    """
    if load_ckpt == '':
        return True
    else:
        if not os.path.isfile(load_ckpt):
            raise ValueError('Argument load_ckpt is not a file')
        if not (load_ckpt.endswith('.pt') or load_ckpt.endswith('.pth')):
            raise ValueError('Argument load_ckpt must be a PyTorch model file with extension ".pt" or ".pth"')
    return True

def check_args_optimizer(optimizer):
    """
    Ensures that the optimizer argument is valid. Immediately raises an error if found.
    :param optimizer: The optimizer argument supplied as an argument of the main program.
    :return: True if valid.
    """
    if optimizer not in ['sgd', 'adam']:
        raise ValueError('Argument optimizer is unsupported')
    return True

def check_config(config):
    """
    Ensures that the main program's configuration dictionary is valid. Immediately raises an error if found.
    :param config: Configuration dictionary.
    :return: True if valid.
    """
    supported_modalities = ['audio', 'text']

    # 1st (outermost) layer
    if "name" not in config:
        raise ValueError('Configuration is missing "name" key')
    if "data" not in config:
        raise ValueError('Configuration is missing "data" key')

    # 2nd layer
    for modality in config['data']:
        if modality not in supported_modalities:
            raise ValueError('Configuration contains unsupported modality under "data" key: ' + modality + '')

        # 3rd layer
        if "featurizer" not in config['data'][modality]:
            raise ValueError('Configuration is missing "featurizer" key under "data"-->"' + modality + '" key')

        # 4th layer
        if "name" not in config['data'][modality]['featurizer']:
            raise ValueError('Configuration is missing "name" key under "data"-->"' + modality + '"-->"featurizer" key')

    return True
