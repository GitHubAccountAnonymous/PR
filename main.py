import copy
from data.dataset import MODEL2DATASET
import logging
from models import SUPPORTED_MODELS
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.eval import display_results, prf1
from utils.inspect import SUPPORTED_RESTORATION_FXNS
from utils.misc import fix_label_dims
from utils.os import prep_program

logger = logging.getLogger(__name__)


def main(args, config):
    model = SUPPORTED_MODELS[config['name']](config['model'])
    if args.load_ckpt != '':
        logging.info('Loading saved checkpoint ' + args.load_ckpt)
        # model.load_state_dict(torch.load(args.load_ckpt, map_location=torch.device(args.device)))
        model = torch.load(args.load_ckpt, map_location=torch.device(args.device))
    logging.info(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    model = model.to(args.device)
    dataset = MODEL2DATASET[config['name']]

    if args.mode == 'train':
        train_dataset = dataset(args.data_path, config['data'], 'train', args.featurize)
        dev_dataset = dataset(args.data_path, config['data'], 'dev', args.featurize)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        model = train(model, train_loader, dev_loader, args)

    if args.mode == 'train' or args.mode == 'predict':
        test_dataset = dataset(args.data_path, config['data'], 'test-amb', args.featurize)
    elif args.mode == 'inspect':
        test_dataset = dataset(args.data_path, config['data'], 'inspect', args.featurize)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_results, _ = predict(model, test_loader, args)
    display_results(test_results)


def train(model, train_loader, dev_loader, args):
    """
    Trains a model.
    :param model: Model to train.
    :param train_loader: DataLoader object for training set.
    :param dev_loader: DataLoader object for validation set.
    :param args: Main program arguments.
    :return: Fully trained model.
    """
    criterion = nn.CrossEntropyLoss(reduction='sum')
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best = 0
    open('run-metric.log', 'w').close()

    logging.info('Starting training')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        data_count = 0

        for i, data in enumerate(train_loader, 0):
            try:
                inputs = data['Input'].to(args.device)
                labels = data['Label'].to(args.device)
            # AttributeError occurs when attempting to move a list to device, and such list is returned when data sample
            # is invalid
            except AttributeError:
                continue

            labels = fix_label_dims(labels)
            # labels.shape should be [B]
            data_count += args.batch_size
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                logging.info('Epoch ' + str(epoch + 1) + ' | ' +
                             'Processing sample ' + str(data_count) + ' / ' + str(len(train_loader.dataset)) + ' | ' +
                             'Loss/sample: ' + str(float(loss) / args.batch_size))
            if i % args.save_freq == 0:
                torch.save(model, os.path.join(args.save_path, 'last.pt'))

        torch.save(model, os.path.join(args.save_path, 'last.pt'))
        logging.info('End of epoch ' + str(epoch+1) + ' | Epoch loss: ' + str(total_loss))

        results, primary_metric = predict(model, dev_loader, args)
        display_results(results)
        to_write = 'Epoch ' + str(epoch+1) + ': ' + str(primary_metric)
        if primary_metric > best:
            best = primary_metric
            torch.save(model, os.path.join(args.save_path, 'best.pt'))
            to_write += ' (written as best.pt)'
        with open('run-metric.log', 'a') as f:
            f.write(to_write + '\n')

    return model

def predict(model, loader, args):
    """
    Makes model predictions using PyTorch.
    :param model: Model to use for predictions.
    :param loader: DataLoader object for dataset to predict on.
    :param args: Main program arguments.
    :return: (`results`, `primary_metric`). `results` is a dictionary with keys `'comma'`, `'fs'` (full stop), `'qm'`
    (question mark), and `'overall'`, and each value being a tuple (precision, recall, f1). `primary_metric` is the
    primary metric within `results` used to compare models.
    """
    model.eval()
    data_count = 0
    all_pred = []
    all_lab = []
    if args.mode == 'predict':
        logging.info('Predicting')
    elif args.mode == 'inspect':
        logging.info('Restoring punctuation for inspection')
        restore_punct = SUPPORTED_RESTORATION_FXNS[type(model).__name__]
        all_ident = []

    for i, data in enumerate(loader, 0):
        if i % 10 == 0:
            logging.info('Processing sample ' + str(data_count) + ' / ' + str(len(loader.dataset)))
        try:
            inputs = data['Input'].to(args.device)
        # AttributeError occurs when attempting to move a list to device, and list is returned when data sample
        # is invalid
        except AttributeError:
            continue

        labels = data['Label']
        labels = fix_label_dims(labels)
        # labels.shape should be [B]
        data_count += args.batch_size

        outputs = model(inputs)
        _, pred = torch.max(outputs, dim=1)

        all_pred.append(copy.deepcopy(pred.detach().cpu()))
        all_lab.append(copy.deepcopy(labels.detach().cpu()))
        if args.mode == 'inspect':
            all_ident.append(data['Ident'])

    if args.mode == 'inspect':
        texts = restore_punct(all_pred, all_ident)
        for text in texts:
            logging.info(text + '\n')

    results = prf1(np.concatenate(all_lab), np.concatenate(all_pred), percent=True)
    return results, results['overall'][2]


def add_arguments(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config_path', type=str, default='configs/EfficientPunctTDNN.json')
    parser.add_argument('--data_path', type=str, default='data/mustcv1/')
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for PyTorch DataLoader')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100000, help='Number of training steps per model save')
    parser.add_argument('--featurize', action='store_true', help='Whether or not to (re)generate features')
    return parser


if __name__ == "__main__":
    args, config = prep_program(add_arguments, os.path.splitext(os.path.basename(__file__))[0])
    gpu = ['BERTMLP']
    for modality in config['data']:
        if config['data'][modality]['featurizer']['name'] in gpu:
            config['data'][modality]['featurizer']['device'] = args.device
    main(args, config)
