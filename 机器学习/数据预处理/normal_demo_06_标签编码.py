#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  normal_demo_06_标签编码.py
:time  2024/6/13 10:35
:desc  
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    ["audi", "ford", "audi", "bmw", "ford", "bmw"]
)

lb_encoder = sp.LabelEncoder()
lb_samples = lb_encoder.fit_transform(raw_samples)
print(lb_samples)
print(lb_encoder.inverse_transform(lb_samples))
