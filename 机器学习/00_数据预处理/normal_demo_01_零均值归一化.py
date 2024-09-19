#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_01_零均值归一化.py
:time  2024/6/13 9:26
:desc  零均值归一化-->特征归一化：将特征转为均值为0，标准差为1
"""
import numpy as np
import sklearn.preprocessing as sp

# 样本数据
raw_samples = np.array(
    [[3.0, -1.0, 2.0],
     [0.0, 4.0, 3.0],
     [1.0, -4.0, 2.0]]
)
print(raw_samples)
mean_value = raw_samples.mean(axis=0)
std_value = raw_samples.std(axis=0)
print(mean_value,std_value)


# 求每一列的均值和标准差
mean_value = raw_samples.mean(axis=0)
std_value = raw_samples.std(axis=0)
print(mean_value, std_value)
std_samples = raw_samples.copy()
for col in std_samples.T:
    col_mean = col.mean()
    col_std = col.std()
    col -= col_mean
    col /= col_std

print(std_samples)
print(std_samples.mean(axis=0))
print(std_samples.std(axis=0))

