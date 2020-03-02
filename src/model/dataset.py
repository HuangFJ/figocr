# -*- coding: utf8 -*-
import string
import random
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
import cv2
from model.vocabulary import vocabulary
from procedure.image import image_resize
from torch.utils.data import Dataset, DataLoader


class OCRDataset(Dataset):

    def __init__(self, train, model, target_length=8):
        super(OCRDataset, self).__init__()

        self.model = model
        self.targetPad = target_length
        self.items = []

        # mnist 数据集
        # root='mnist_png/training'
        # if not train:
        #     root='mnist_png/testing'
        # for i in range(10):
        #     filenames = glob.glob(osp.join(root, str(i), '*.png'))
        #     for fn in filenames:
        #         self.items.append((fn, str(i)))

        # 本地数据集
        # root='drive/My Drive/cv/images/dataset/labels.txt'
        # line = 0
        # with open(root) as f:
        #     for l in f:
        #         line += 1
        #         if train:
        #             if line % 6 == 0:
        #                 continue
        #         else:
        #             if line % 6 != 0:
        #                 continue
                
        #         fn, label = l.strip().split('\t')
        #         fn = osp.join('drive/My Drive/cv/images/dataset/images', fn)
        #         self.items.append((fn, label))

        random.shuffle(self.items)

    def normalize(self, origin):
        # origin 是 numpy 形式的图片 (height, width, channel) 值 0-255，零值背景
        # width 随意；固定 height 是32的
        thumbH = self.model.imgH
        thumbW = int((thumbH / origin.shape[0]) * origin.shape[1])
        # print(thumbH,thumbW)
        
        # 等比缩放
        origin = image_resize(origin, height=thumbH)
        # 去掉原通道
        origin = origin[:, :, 0]

        # 设置宽度
        paddingW = False
        if self.model.imgW is not None:
            thumbW = self.model.imgW
            paddingW = True
        elif thumbW % self.model.downsampleW != 0:
            thumbW = (thumbW // self.model.downsampleW + 1) * self.model.downsampleW
            paddingW = True
        
        if thumbW < thumbH:
            thumbW = thumbH
            paddingW = True

        if paddingW:
            image = np.zeros((thumbH, thumbW))
            image[0 : origin.shape[0], 0 : origin.shape[1]] = origin
            origin = image

        # 值转为 0-1
        origin = origin.astype(np.float32) / 255
        # 添加1个通道
        origin = np.expand_dims(origin, 2)
        return origin

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        fn, label = self.items[index]
        image = self.normalize(cv2.imread(fn))
        input_length = image.shape[1] // self.model.downsampleW

        if self.targetPad is not None:
            if len(label) > self.targetPad:
                label = label[ : self.targetPad]
            elif len(label) < self.targetPad:
                label += vocabulary[0]*(self.targetPad - len(label))
        
        image = to_tensor(image)
        input_length = torch.full(size=(1,), fill_value=input_length, dtype=torch.long)
        target = torch.tensor([vocabulary.find(x) for x in label], dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=len(label), dtype=torch.long)
    
        return image, target, input_length, target_length