# -*- coding: utf8 -*-
import random
import numpy as np
import torch
import cv2
from model.vocabulary import vocabulary
from torch.utils.data import Dataset
import glob
from pathlib import Path


class OCRDataset(Dataset):

    def __init__(self, train, normalizer, input_length):
        super().__init__()
        self.items = []
        self.normalizer = normalizer
        self.input_length = input_length

        # mnist 数据集
        # !wget https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz?raw=true
        # !mv mnist_png.tar.gz?raw=true mnist_png.tar.gz
        # !tar -xzf mnist_png.tar.gz
        # root = Path('mnist_png/training')
        # if not train:
        #     root = Path('mnist_png/testing')
        # for i in range(10):
        #     filenames = glob.glob(str(root.joinpath(str(i), '*.png')))
        #     for fn in filenames:
        #         self.items.append((fn, str(i)))

        # 本地数据集
        # root=Path('drive/My Drive/cv/images/dataset/labels.txt')
        # line = 0
        # with open(root) as f:
        #     for l in f:
        #         line += 1
        #         if train:
        #             if line % 20 == 0:
        #                 continue
        #         else:
        #             if line % 20 != 0:
        #                 continue

        #         fn, label = l.strip().split('\t')
        #         fn = root.parent.joinpath('images', fn)
        #         self.items.append((str(fn), label))

        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        fn, label = self.items[index]
        target = [vocabulary.find(x) for x in label]

        image = self.normalizer(cv2.imread(fn))
        target = torch.tensor(target, dtype=torch.long)
        input_length = torch.full(
            size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(
            size=(1, ), fill_value=len(target), dtype=torch.long)
        return image, target, input_length, target_length
