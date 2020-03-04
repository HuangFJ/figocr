# -*- coding: utf8 -*-
import string

blank = '-'
# 小数点，数字，大写字母
vocabulary = blank + '.' + string.digits + string.ascii_uppercase

def decode_target(sequence):
    return ''.join([vocabulary[x] for x in sequence]).replace(' ', '')


def decode(sequence):
    a = ''.join([vocabulary[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != blank and x != a[j+1]])
    if len(s) == 0:
        s = blank
    if a[-1] != blank and s[-1] != a[-1]:
        s += a[-1]
    return s.strip(blank)