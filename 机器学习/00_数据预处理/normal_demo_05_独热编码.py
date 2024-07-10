#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_05_独热编码.py
:time  2024/6/13 10:24
:desc  独热编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    [[1, 3, 2],
     [7, 5, 4],
     [1, 8, 6],
     [7, 3, 9]]
)
one_hot_encoder = sp.OneHotEncoder(
    sparse=False,
    dtype="int32",
    categories="auto"
)
oh_samples = one_hot_encoder.fit_transform(raw_samples)
print(oh_samples)
print(one_hot_encoder.inverse_transform(oh_samples))
