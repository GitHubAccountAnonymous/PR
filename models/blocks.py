import torch.nn as nn


class ConvReLUBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        """
        :param in_channels: `int`. Number of input channels.
        :param out_channels: `int`. Number of output channels.
        :param kernel_size: `int`. Kernel size for Conv1d layer.
        :param stride: `int`. Stride for Conv1d layer.
        :param padding: `int`. Padding for Conv1d layer.
        :param dilation: `int`. Dilation for Conv1d layer.
        """
        super(ConvReLUBatchNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        return x


class ConvReLUDropout(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout=0.0):
        """
        :param in_channels: `int`. Number of input channels.
        :param out_channels: `int`. Number of output channels.
        :param kernel_size: `int`. Kernel size for Conv1d layer.
        :param stride: `int`. Stride for Conv1d layer.
        :param padding: `int`. Padding for Conv1d layer.
        :param dilation: `int`. Dilation for Conv1d layer.
        """
        super(ConvReLUDropout, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x