from models.blocks import ConvReLUBatchNorm, ConvReLUDropout
from models.text import EfficientPunctBERT
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientPunct(nn.Module):

    def __init__(self, config):
        super(EfficientPunct, self).__init__()
        self.eptdnn = EfficientPunctTDNN(config)
        self.epbert = EfficientPunctBERT(config)
        self.alpha = 0.4

    def forward(self, x):
        x_tdnn = self.eptdnn(x)
        x_bert = self.epbert(x[:, :768, 150])
        p_tdnn = F.softmax(x_tdnn, dim=1)
        p_bert = F.softmax(x_bert, dim=1)
        p = self.alpha * p_tdnn + (1 - self.alpha) * p_bert
        return p


class EfficientPunctTDNN(nn.Module):

    def __init__(self, config):
        super(EfficientPunctTDNN, self).__init__()
        self.linear0 = nn.Linear(1792, 512)
        self.tdnn = nn.Sequential(
            ConvReLUBatchNorm(in_channels=512, out_channels=256, kernel_size=9),
            ConvReLUBatchNorm(in_channels=256, out_channels=256, kernel_size=9, dilation=2),
            ConvReLUBatchNorm(in_channels=256, out_channels=128, kernel_size=5),
            ConvReLUBatchNorm(in_channels=128, out_channels=128, kernel_size=5, dilation=2),
            ConvReLUBatchNorm(in_channels=128, out_channels=64, kernel_size=7),
            ConvReLUBatchNorm(in_channels=64, out_channels=64, kernel_size=7, dilation=2),
            ConvReLUBatchNorm(in_channels=64, out_channels=4, kernel_size=5)
        )
        self.linear1 = nn.Linear(243, 70)
        self.batchnorm = nn.BatchNorm1d(70)
        self.linear2 = nn.Linear(70, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear0(x)
        x = F.relu(x)
        x = torch.transpose(x, 1, 2)

        x = self.tdnn(x)

        x0 = self.linear1(x[:, 0, :])
        x0 = F.relu(x0)
        x0 = self.batchnorm(x0)

        x1 = self.linear1(x[:, 1, :])
        x1 = F.relu(x1)
        x1 = self.batchnorm(x1)

        x2 = self.linear1(x[:, 2, :])
        x2 = F.relu(x2)
        x2 = self.batchnorm(x2)

        x3 = self.linear1(x[:, 3, :])
        x3 = F.relu(x3)
        x3 = self.batchnorm(x3)

        x0 = self.linear2(x0)
        x1 = self.linear2(x1)
        x2 = self.linear2(x2)
        x3 = self.linear2(x3)

        x = torch.hstack((x0, x1, x2, x3))
        return x


class LengthConditional(nn.Module):

    def __init__(self, config):
        super(LengthConditional, self).__init__()
        self.parts = config['partitions']

    def forward(self, x):
        return x


class UniPunc(nn.Module):

    def __init__(self, config):
        super(UniPunc, self).__init__()
        self.downsample_conv = nn.Sequential(
            ConvReLUDropout(in_channels=1024, out_channels=768, kernel_size=15, stride=5, dropout=0.1),
            ConvReLUDropout(in_channels=768, out_channels=768, kernel_size=15, stride=5, dropout=0.1)
        )

        self.self_attn_1 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.self_attn_2 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.dropout_2 = nn.Dropout(p=0.1)

        self.linear = nn.Linear(768, 4)

    def forward(self, x):
        H_l = x[:, :768, :]
        x_a = x[:, 768:, :]
        H_a = self.downsample_conv(x_a)

        H_l = torch.transpose(H_l, 1, 2)
        H_a = torch.transpose(H_a, 1, 2)

        S_l, _ = self.self_attn_1(H_l, H_l, H_l)
        S_a, _ = self.cross_attn_1(H_l, H_a, H_a)
        H_l = S_l + S_a + H_l
        H_l = self.dropout_1(H_l)

        S_l, _ = self.self_attn_2(H_l, H_l, H_l)
        S_a, _ = self.cross_attn_2(H_l, H_a, H_a)
        H_h = S_l + S_a + H_l
        H_h = self.dropout_2(H_h)

        H_h = torch.mean(H_h, dim=1)
        x = self.linear(H_h)
        x = F.softmax(x, dim=1)
        return x

