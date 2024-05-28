from data.build.builder import SUPPORTED_DATASET_BUILDERS
import logging
import os
from utils.os import prep_program

logger = logging.getLogger(__name__)


def main(args, config):
    builder = SUPPORTED_DATASET_BUILDERS[config['name']](args.data_path, config)
    builder.build()


def add_arguments(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config_path', type=str, default='configs/YouTube.json')
    parser.add_argument('--data_path', type=str, default='data/custom/')
    return parser


if __name__ == "__main__":
    args, config = prep_program(add_arguments, os.path.splitext(os.path.basename(__file__))[0])
    config['device'] = args.device
    main(args, config)
