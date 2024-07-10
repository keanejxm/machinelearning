#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_04_二值化.py
:time  2024/6/13 10:16
:desc  二值化：根据一个事先给定的阈值，用0和1来表示特征值
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    [[65.5, 89.0, 73.0],
     [55.0, 99.0, 98.5],
     [45.0, 22.5, 60.0]]
)
bin_samples = raw_samples.copy()
mask1 = bin_samples < 60
mask2 = bin_samples >= 60

bin_samples[mask1] = 0
bin_samples[mask2] = 1
print(bin_samples)

bin_1 = sp.Binarizer(threshold=59)
bin_samples = bin_1.transform(raw_samples)
print(bin_samples)