#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  min_max_scale.py
:time  2023/12/5 13:56
:desc  最大值最小值归一化
"""
import numpy as np
import sklearn.preprocessing as sp

raw_sample = np.array([[3.0, -100.0, 2000.0],
                       [0.0, 400.0, 3000.0],
                       [1.0, -400.0, 2000.0]])

mms_sample = raw_sample.copy()

for col in mms_sample.T:
    col_max = col.max()
    col_min = col.min()
    col -= col_min
    col /= (col_max - col_min)
print(mms_sample)

# res = sp.minmax_scale(raw_sample)
# print(res)
mms = sp.MinMaxScaler()
mms.fit(raw_sample)
res = mms.transform(raw_sample)
print(res)