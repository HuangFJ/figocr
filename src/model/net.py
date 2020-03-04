# -*- coding: utf8 -*-
import torch.nn as nn
from collections import OrderedDict
import torch


class OCRNet(nn.Module):
    def __init__(self, n_classes, input_shape=(1, 32, 320)):
        super(OCRNet, self).__init__()
        self.input_shape = input_shape

        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]

        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = self.input_shape[0]
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)

        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        # encoder
        self.cnn = nn.Sequential(modules)
        # rnn
        self.lstm = nn.LSTM(input_size=self.infer_features(
        ), hidden_size=128, num_layers=2, bidirectional=True)
        # dense
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        # N C H W  => N 1 32/(2*2*2*2*2) 320/(2*2*2*2*1) => N 1 1 20
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # N C*H W  => N 1 20
        return x.shape[1]  # C*H  => 1

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)  # W N C*H  => 20 N 1
        # W N num_directions∗hidden_size  => 20 N 2*128 => 20 N 256
        x, _ = self.lstm(x)
        x = self.fc(x)  # W N n_classes  => 20 N n_classes
        return x
