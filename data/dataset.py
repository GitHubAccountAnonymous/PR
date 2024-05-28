from abc import abstractmethod
from feats import SUPPORTED_FEATURIZERS
import logging
import numpy as np
import os
import pickle
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from utils.misc import sample
from utils.os import list_visible, read, read_lines
from utils.text import fix_formatting, get_non_alphanum, get_punctuation, remove_special

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(self, path, config, split):
        self.path = path
        self.config = config
        self.split = split
        self.egs = []

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.egs)


class PostKaldiConcatDataset(BaseDataset):

    def __init__(self, path, config, split, featurize=True):
        """
        :param path: `str`. Path to dataset, which should conform to the standard data format (see README file).
        :param config: `dict`. Configuration dictionary under `"data"`.
        :param split: `str`. `'train'`, `'dev'`, or `'test'`.
        :param featurize: `bool`. Whether to generate features.
        """
        super().__init__(path, config, split)
        if featurize:
            self._featurize(self.split)
        self.egs = self._read_examples(self.split)
        if self.split == 'train':
            self.egs = self._balance_classes(self.egs)
        self.off_center = int(self.config['audio']['context'] / 2)

    def __getitem__(self, idx):
        eg = self.egs[idx]
        ident = {'UttID': eg[0], 'Start': eg[1], 'Path': os.path.join(self.path, self.split)}
        if self.split == 'train':
            # A training example is allowed to be missing, in which case we randomly sample another one.
            # However, a validation or test example is not allowed to be missing.
            try:
                f = open(os.path.join(self.path, 'feat', self.split, 'concat', 'egs', eg[0] + '.feat'), 'rb')
            except FileNotFoundError:
                return self.__getitem__(random.randint(0, len(self.egs) - 1))
        else:
            f = open(os.path.join(self.path, 'feat', self.split, 'concat', 'egs', eg[0] + '.feat'), 'rb')
        full = np.transpose(pickle.load(f))
        f.close()

        start = eg[1]
        end = start + 2 * self.off_center + 1  # non-inclusive
        label = eg[2]

        if start < 0:
            left_pad = np.zeros((full.shape[0], abs(start)))
            start = 0
        else:
            left_pad = np.zeros((full.shape[0], 0))

        if end > full.shape[1]:
            right_pad = np.zeros((full.shape[0], end - full.shape[1]))
            end = full.shape[1]
        else:
            right_pad = np.zeros((full.shape[0], 0))

        eg = torch.from_numpy(np.hstack((left_pad, full[:, start:end], right_pad))).type(torch.float32)
        assert eg.shape == (full.shape[0], 2 * self.off_center + 1)

        item = {'Input': eg, 'Label': label}
        if self.split == 'inspect':
            item['Ident'] = ident
        return item

    def _balance_classes(self, egs):
        """
        Balances classes by oversampling the minority classes.
        :param egs: `list` of examples needed to be balanced.
        :return: `list` of examples with balanced classes.
        """
        egs_by_class = [[] for _ in range(4)]
        for eg in egs:
            if eg[2] == 0:
                egs_by_class[0].append(eg)
            elif eg[2] == 1:
                egs_by_class[1].append(eg)
            elif eg[2] == 2:
                egs_by_class[2].append(eg)
            elif eg[2] == 3:
                egs_by_class[3].append(eg)

        most_freq = np.argmax([len(l) for l in egs_by_class])
        max_count = len(egs_by_class[most_freq])

        for i in set(range(4)) - {most_freq}:
            additional = sample(egs_by_class[i], max_count - len(egs_by_class[i]))
            egs_by_class[i].extend(additional)
        return [eg for egs_class in egs_by_class for eg in egs_class]

    def _featurize(self, split):
        self.featurizer = SUPPORTED_FEATURIZERS[self._get_featurizer_names(self.config)](self.config)
        logging.info('Featurizing ' + split + ' set')
        self.featurizer(os.path.join(self.path, split))

    def _get_featurizer_names(self, config):
        """
        Obtains featurizer names.
        :param config: `dict`. Configuration dictionary under `"data"`.
        :return: `tuple` in the format (audio, text), where each element is either a string or None, if the
        corresponding modality is missing.
        """
        modalities = ['audio', 'text']
        return tuple([config[m]['featurizer']['name'] if m in config else None for m in modalities])

    def _read_examples(self, split):
        egs = read_lines(os.path.join(self.path, 'feat', split, 'concat', 'egs_txt', 'egs.txt'))
        for i in range(len(egs)):
            eg = egs[i].split()
            eg[1] = int(eg[1])
            eg[2] = int(eg[2])
            egs[i] = eg
        return egs


class PostKaldiConcatTextDataset(PostKaldiConcatDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        input = item['Input'][:768, 150]
        item['Input'] = input
        return item


class TextDataset(BaseDataset):
    def __init__(self, path, config, split, featurize=None):
        """
        :param path: `str`. Path to dataset, which should conform to the standard data format (see README file).
        :param config: `dict`. Configuration dictionary under `"data"`.
        :param split: `str`. `'train'`, `'dev'`, or `'test'`.
        :param featurize: `NoneType`. This parameter is ignored in this class, as featurization is mandatory. The
        BERTTokenizerFeaturizer is used.
        """
        super().__init__(path, config, split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.feat_config = self.config['text']['featurizer']
        self.text_path = os.path.join(self.path, self.split, 'text')
        files = [file for file in list_visible(self.text_path) if file.endswith('.txt')]
        self.texts = []
        logging.info('Loading text dataset')
        for file in tqdm(files):
            with open(os.path.join(self.text_path, file), 'r') as f:
                text = f.read().strip()
            self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        if sentence == '':
            return {'Input': [], 'Label': [], 'Ident': {'Text': ''}}

        L = len(sentence)
        check_start = 0
        while not sentence[check_start].isalnum():
            check_start += 1
            if check_start >= L:
                # In this case, the entire sentence is non-alphanumeric
                return {'Input': [], 'Label': [], 'Ident': {'Text': ''}}
        sentence = sentence[check_start:]
        sentence = fix_formatting(sentence)

        unpunct = sentence.lower()
        marks = get_non_alphanum(unpunct)
        for p in marks:
            if p != "'":
                unpunct = unpunct.replace(p, "")

        inputs = unpunct
        labels = get_punctuation(sentence)
        if len(labels) == 0:
            return {'Input': [], 'Label': [], 'Ident': {'Text': unpunct}}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs_list = inputs.split()
        inputs_word_tokens = tokenizer.tokenize(inputs)
        inputs_num_tokens = tokenizer(inputs, return_tensors='pt')
        for k in inputs_num_tokens:
            inputs_num_tokens[k] = torch.squeeze(inputs_num_tokens[k])

        labels_tokenized = []
        checking = False
        for j in range(len(inputs_word_tokens)):
            word_token = inputs_word_tokens[j].replace('#', '')
            if word_token == inputs_list[0]:
                try:
                    labels_tokenized.append(labels.pop(0))
                # This is some sort of invalid data sample with unresolvable formatting
                except KeyError:
                    continue
                inputs_list.pop(0)
            else:
                if not checking:
                    checking = True
                    check_str = word_token
                    check_counter = 1
                else:
                    check_str += word_token
                    check_counter += 1
                    if check_str == inputs_list[0]:
                        try:
                            lab = labels.pop(0)
                        # This is some sort of invalid data sample with unresolvable formatting
                        except KeyError:
                            continue
                        lab_seq = [0 for _ in range(check_counter)]
                        lab_seq[-1] = lab
                        labels_tokenized += lab_seq
                        inputs_list.pop(0)
                        checking = False

        labels_tokenized = torch.Tensor(labels_tokenized).to(torch.int64)
        to_return = {'Input': inputs_num_tokens, 'Label': labels_tokenized, 'Ident': {'Text': unpunct}}
        return to_return


MODEL2DATASET = {
    'BERTMLP': TextDataset,
    'EfficientPunct': PostKaldiConcatDataset,
    'EfficientPunctBERT': PostKaldiConcatTextDataset,
    'EfficientPunctTDNN': PostKaldiConcatDataset,
    'UniPunc': PostKaldiConcatDataset
}