#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_03_样本归一化.py
:time  2024/6/13 10:02
:desc  样本归一化-->
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    [[10.0, 20.0, 5.0],
     [8.0, 10.0, 1.0]]
)
nor_samples = raw_samples.copy()
for row in nor_samples:
    row /= (abs(row).sum())
print(nor_samples)

print(sp.normalize(raw_samples, norm="l1"))  # l1范数
print(sp.normalize(raw_samples, norm="l2"))  # l2范数
