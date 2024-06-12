#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pd_demo_04_算术平均值、加权平均值、最值、中位数.py
:time  2024/6/12 14:22
:desc  算术平均值、加权平均值、最值、中位数、标准差
"""
import pandas as pd
import numpy as np

# 平均值
w = np.array([3, 1, 1, 1, 1, 1, 1])
print(np.average(w))
# 最值
a = np.random.randint(10, 100, 9)
print(a)
print(np.max(a), np.min(a), np.ptp(a))

print(np.argmax(a),np.argmin(a))
