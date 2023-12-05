#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  scale_demo_01.py
:time  2023/12/5 11:19
:desc  均值移除（标准移除）:每列平均值变为0，标准差变为1
"""
import sklearn.preprocessing as sp

import numpy as np

raw_sample = np.array([[3.0, -100.0, 2000.0],
                       [0.0, 400.0, 3000.0],
                       [1.0, -400.0, 2000.0]])
std_sample = raw_sample.copy()
print(raw_sample[0])
# 当前的数据-当前的平均值，得到离差
# 离差/当前列的标准差
for col in std_sample.T:
    mean_val = col.mean()
    std_val = col.std()
    col -= mean_val
    col /= std_val
print(std_sample)

res = sp.scale(raw_sample)
print(res)
# axis = 0 axis：轴向 等于0 列方向，等于1 行方向
print(res.mean(axis=0))
print(res.std(axis=0))
