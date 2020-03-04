# -*- coding: utf8 -*-
from model.net import OCRNet
from model.dataset import OCRDataset
from model.vocabulary import decode, decode_target, vocabulary
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from procedure.image import cv2_imshow
import logging
from tensorboardX import SummaryWriter
from procedure.image import image_resize
from torchvision.transforms.functional import to_tensor


def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


class Normalizer(object):

    def __init__(self, imgH, imgW):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, origin):
        # origin 是 numpy 形式的图片 (height, width, channel) 值 0-255，零值背景
        thumbW = int((self.imgH / origin.shape[0]) * origin.shape[1])
        
        # 等比缩放
        origin = image_resize(origin, height=self.imgH)
        # 去掉原通道
        origin = origin[:, :, 0]
        # 值转为 0-1
        origin = torch.FloatTensor(origin) / 255

        padding_size = self.imgW - thumbW
        # 设置宽度
        origin = F.pad(origin, ((padding_size + 1) // 2, padding_size // 2))

        # 添加1个通道
        origin = origin.unsqueeze(0) # 1 H W
        
        return origin


class OCRModel(object):

    input_shape=(1, 32, 320) # C H W
    n_classes = len(vocabulary)
    batch_size = 128


    def __init__(self):
        super().__init__()

        self.optimizer_state_dict = None

        self.net = OCRNet(self.n_classes, self.input_shape)
        self.criterion = nn.CTCLoss()

        self.writer = SummaryWriter(f'logs/OCRModel')
        self.writer.add_graph(self.net, torch.zeros((1,) + self.input_shape))

        self.normalizer = Normalizer(self.input_shape[1], self.input_shape[2])

        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.criterion = self.criterion.cuda()
        
        self.net = torch.nn.DataParallel(self.net)


    def train(self, optimizer, epoch, dataloader):
        self.net.train()
        loss_mean = 0
        acc_mean = 0
        with tqdm(dataloader) as pbar:
            for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = self.net(data)
                
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = self.criterion(output_log_softmax, target, input_lengths, target_lengths)
                
                loss.backward()
                optimizer.step()

                loss = loss.item()
                acc = calc_acc(target, output)
                
                iteration = (epoch - 1) * len(dataloader) + batch_index
                self.writer.add_scalar('train/loss', loss, iteration)
                self.writer.add_scalar('train/acc', acc, iteration)
                self.writer.add_scalar('train/error_rate', 1 - acc, iteration)
                
                if batch_index == 0:
                    loss_mean = loss
                    acc_mean = acc
                
                loss_mean = 0.1 * loss + 0.9 * loss_mean
                acc_mean = 0.1 * acc + 0.9 * acc_mean
                
                pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
        
        return loss_mean, acc_mean


    def valid(self, epoch, dataloader):
        self.net.eval()
        with tqdm(dataloader) as pbar, torch.no_grad():
            loss_sum = 0
            acc_sum = 0
            for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                
                output = self.net(data)
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = self.criterion(output_log_softmax, target, input_lengths, target_lengths)
                
                loss = loss.item()
                acc = calc_acc(target, output)
                
                loss_sum += loss
                acc_sum += acc
                
                loss_mean = loss_sum / (batch_index + 1)
                acc_mean = acc_sum / (batch_index + 1)
                
                pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

        return loss_mean, acc_mean
    

    def collate_fn(self, batch):
        batch.sort(key=lambda data: len(data[1]), reverse=True)
        bat_d = []
        bat_t = []
        bat_i_lengths = []
        bat_t_lengths = []
        for d, t, i_length, t_length in batch:
            bat_d.append(d.numpy())
            bat_t.append(t)
            bat_i_lengths.append(i_length)
            bat_t_lengths.append(t_length)

        bat_t = torch.nn.utils.rnn.pad_sequence(bat_t, batch_first=True)

        bat_d = torch.Tensor(bat_d)
        bat_t = torch.LongTensor(bat_t)
        bat_i_lengths = torch.IntTensor(bat_i_lengths)
        bat_t_lengths = torch.IntTensor(bat_t_lengths)

        return bat_d, bat_t, bat_i_lengths, bat_t_lengths


    def evolution(self, epochs=10, lr=1e-4, optimizer=None):
        train_set = OCRDataset(True, self.normalizer, self.input_shape[2] // (2**4))
        valid_set = OCRDataset(False, self.normalizer, self.input_shape[2] // (2**4))

        train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=2, collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, num_workers=2, collate_fn=self.collate_fn)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train(optimizer, epoch, train_loader)
            valid_loss, valid_acc = self.valid(epoch, valid_loader)
            
            self.writer.add_scalars('epoch/loss', {'train': train_loss, 'valid': valid_loss}, epoch)
            self.writer.add_scalars('epoch/acc', {'train': train_acc, 'valid': valid_acc}, epoch)
            self.writer.add_scalars('epoch/error_rate', {'train': 1 - train_acc, 'valid': 1 - valid_acc}, epoch)
        
        self.optimizer_state_dict = optimizer.state_dict()
    

    def save_checkpoint(self, checkpoint_path):
        # - for a model, maps each layer to its parameter tensor;
        # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
        state = {
            'state_dict': self.net.state_dict(),
            'optimizer' : self.optimizer_state_dict
        }
        torch.save(state, checkpoint_path)
        logging.info('model saved to %s' % checkpoint_path)


    def load_checkpoint(self, checkpoint_path):
        if torch.cuda.is_available():
            state = torch.load(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(state['state_dict'])
        self.optimizer_state_dict = state['optimizer']
        logging.info('model loaded from %s' % checkpoint_path)


    def predict(self, image):
        self.net.eval()
        image = self.normalizer(image).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self.net(image)
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        logging.info(''.join([vocabulary[x] for x in output_argmax[0]]))
        return decode(output_argmax[0])