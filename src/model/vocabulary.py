# -*- coding: utf8 -*-
import string
import numpy as np


vocabulary = '-.' + string.digits + string.ascii_uppercase


def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


def decode_target(sequence):
    return ''.join([vocabulary[x] for x in sequence]).replace(' ', '')


def decode(sequence):
    a = ''.join([vocabulary[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != vocabulary[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != vocabulary[0] and s[-1] != a[-1]:
        s += a[-1]
    return s