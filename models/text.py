import torch
import torch.nn as nn
from transformers import BertModel


class EfficientPunctBERT(nn.Module):

    def __init__(self, config):
        super(EfficientPunctBERT, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class BERTMLP(nn.Module):

    def __init__(self, config):
        """
        :param config: `dict`. Configuration dictionary with a `"linear"` key to indicate dimensions of linear layer(s).
        """
        super(BERTMLP, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.mlp = nn.Sequential()
        for i in range(len(self.config['linear']) - 1):
            self.mlp.append(nn.Linear(self.config['linear'][i], self.config['linear'][i+1]))
            if i < len(self.config['linear']) - 2:
                self.mlp.append(nn.ReLU())

    def bert_last_hidden(self, x):
        return self.bert(**x).last_hidden_state[:, 1:-1, :]

    def forward(self, x):
        # x['input_ids'].shape should be [1, N]
        x = self.bert_last_hidden(x)
        # x.shape should be [1, N-2, 768]
        x = torch.squeeze(x, dim=0)
        x = self.mlp(x)
        # x.shape should be [N-2, 4]
        return x