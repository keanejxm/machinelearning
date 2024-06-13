#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_02_线型函数归一化.py
:time  2024/6/13 9:41
:desc  线型函数归一化-->特征归一化
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]]
).astype("float64")
for col in raw_samples.T:
    col_max = col.max()
    col_min = col.min()
    col -= col_min
    col /= (col_max - col_min)
print(raw_samples)

mms = sp.MinMaxScaler()
mms.fit(raw_samples)
res = mms.transform(raw_samples)
print(res)