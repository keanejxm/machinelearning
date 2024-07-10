#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  binary.py
:time  2023/12/5 14:15
:desc  二值化：大于阈值的转为1，小于阈值的转为0
"""
import numpy as np
import sklearn.preprocessing as sp
raw_sample = np.array([[66.6, 88.8, 55.5],
                       [12.3, 45.6, 78.9],
                       [74.1, 85.2, 36.9]])
bin_sample = raw_sample.copy()
# bin_sample[bin_sample <= 60.0]=0
# bin_sample[bin_sample >60.0] = 1
# print(bin_sample)
res = np.where(bin_sample>60,1,0)
print(res)

res = sp.binarize(raw_sample,threshold=60)
print(res)
bin_normal = sp.Binarizer(threshold=60)
res = bin_normal.transform(raw_sample)
print(res)