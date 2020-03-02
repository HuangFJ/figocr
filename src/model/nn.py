# -*- coding: utf8 -*-
import torch.nn as nn
from collections import OrderedDict
import torch
from pathlib import Path


class OCRModel(nn.Module):
    
    def __init__(self, nClasses, imgShape=(1, 32, 320)):
        super(OCRModel, self).__init__()

        self.imgC, self.imgH, self.imgW = imgShape
        self.downsampleH = 1
        self.downsampleW = 1

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
        
        last_channel = self.imgC
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
            if isinstance(k_pool, tuple):
                self.downsampleH *= k_pool[0]
                self.downsampleW *= k_pool[1]
            else:
                self.downsampleH *= k_pool
                self.downsampleW *= k_pool

        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        
        # encoder
        self.cnn = nn.Sequential(modules)

        assert self.imgH % self.downsampleH == 0
        # rnn
        self.lstm = nn.LSTM(input_size=(self.imgH // self.downsampleH) * last_channel, 
                            hidden_size=128, num_layers=2, bidirectional=True)
        # dense
        self.fc = nn.Linear(in_features=256, out_features=nClasses)

    def forward(self, x):
        x = self.cnn(x)
        # b c h w
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def load_state(self):
        model = torch.load(Path(__file__).parent.joinpath('ocr1.pth'), map_location=torch.device('cpu'))
        self.load_state_dict(model.module.state_dict())