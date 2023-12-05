#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normlize_demo.py
:time  2023/12/5 14:07
:desc  归一化：每一行数值转为占比
"""
import numpy as np
import sklearn.preprocessing as sp
raw_sample = np.array([[10.0, 20.0, 5.0],
                       [8.0, 10.0, 1.0]])

abs_sample = raw_sample.copy()
for row in abs_sample:
    row /= abs(row).sum()
print(abs_sample)

res = sp.normalize(raw_sample,norm="l1")
print(res)