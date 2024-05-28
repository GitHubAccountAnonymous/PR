from feats.base import BaseFeaturizer
import logging
from models.text import BERTMLP
import torch
from transformers import BertTokenizer
from utils.os import load_config

logger = logging.getLogger(__name__)


class BERTMLPFeaturizer(BaseFeaturizer):

    def __init__(self, config):
        """
        :param config: `dict`. Configuration dictionary for `"BERTMLP"` under `"featurizer"`.
        """
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_config = load_config(self.config['config'])
        self.model = BERTMLP(self.model_config)
        if 'pretrained' in self.model_config:
            logging.info('Loading ' + self.model_config['pretrained'] + ' for ' + type(self).__name__)
            # self.model.load_state_dict(torch.load(self.model_config['pretrained']))
            self.model = torch.load(self.model_config['pretrained'])
        self.model = self.model.to(self.config['device'])
        self.model.eval()

    def __call__(self, text):
        """
        :param text: `str` of text.
        """
        words_list = text.split()
        inputs_word_tokens = self.tokenizer.tokenize(text)
        inputs_num_tokens = self.tokenizer(text, return_tensors='pt')
        inputs_num_tokens = inputs_num_tokens.to(self.config['device'])
        N = inputs_num_tokens['input_ids'].shape[1]

        if N > 512:
            inputs_num_tokens = inputs_num_tokens['input_ids'][:, 1:-1]
            # List containing indices of split locations. Each element is index of first token in a group.
            split_loc = [0]
            while inputs_num_tokens.shape[1] - split_loc[-1] > 510:
                for i in range(split_loc[-1] + 510, split_loc[-1], -1):
                    if inputs_word_tokens[i] in words_list:
                        break
                split_loc.append(i)
            assert len(split_loc) >= 2

            bert = torch.zeros([N-2, 768], dtype=torch.float32)

            for i in range(len(split_loc)):
                begin = split_loc[i]

                if i != len(split_loc) - 1:
                    end = split_loc[i + 1]
                else:
                    end = inputs_num_tokens.shape[1]

                current_input_ids = torch.hstack((torch.tensor([[101]]).to(self.config['device']),
                                                  inputs_num_tokens[:, begin:end],
                                                  torch.tensor([[102]]).to(self.config['device'])))
                current_inputs_num_tokens = {'input_ids': current_input_ids,
                                             'token_type_ids': torch.zeros((1, current_input_ids.shape[1]),
                                                                           dtype=torch.int64).to(self.config['device']),
                                             'attention_mask': torch.ones((1, current_input_ids.shape[1]),
                                                                          dtype=torch.int64).to(self.config['device'])}
                current_bert = self.model.bert_last_hidden(current_inputs_num_tokens)
                current_bert = torch.squeeze(current_bert, dim=0)
                bert[begin:end, :] = current_bert
            assert end == bert.shape[0]

        else:
            bert = self.model.bert_last_hidden(inputs_num_tokens)
            bert = torch.squeeze(bert, dim=0)

        word_token = 0
        token_groups = []
        for word in words_list:
            span = ''
            tokens = []
            while span != word:
                span += inputs_word_tokens[word_token].replace('#', '')
                tokens.append(word_token)
                word_token += 1
            token_groups.append(tokens)

        assert len(token_groups) == len(words_list)
        assert token_groups[-1][-1] == bert.shape[0] - 1

        # Elements of bert_word_embeddings contain BERT embeddings for each word
        bert_word_embeddings = []
        for group in token_groups:
            if len(group) == 1:
                bert_word_embeddings.append(bert[group[0], :].detach().cpu().numpy())
            elif len(group) > 1:
                assert max(group) == group[-1]
                bert_word_embeddings.append(bert[group[-1], :].detach().cpu().numpy())
            else:
                raise RuntimeError('Length of token group for a word is less than 1')

        return bert_word_embeddings


class BERTTokenizerFeaturizer(BaseFeaturizer):

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, text):
        """
        :param text: `str` of text.
        """
        words_list = text.split()
        inputs_word_tokens = self.tokenizer.tokenize(text)
        inputs_num_tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        N = inputs_num_tokens['input_ids'].shape[1]
        if N > 512:
            raise RuntimeError('Text contains more than 512 tokens, which is unsupported by ' + type(self).__name__)

        word_token = 0
        token_groups = []
        for word in words_list:
            span = ''
            tokens = []
            while span != word:
                span += inputs_word_tokens[word_token].replace('#', '')
                tokens.append(word_token)
                word_token += 1
            token_groups.append(tokens)

        print('#################################')
        print(inputs_word_tokens, inputs_num_tokens)
        print('')
        print(token_groups)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('')

        assert len(token_groups) == len(words_list)
        # assert token_groups[-1][-1] == bert.shape[0] - 1

SUPPORTED_TEXT_FEATURIZERS = {
    'BERTMLP': BERTMLPFeaturizer,
}