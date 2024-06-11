#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_4_ndarray数据维度操作.py
:time  2024/6/11 14:58
:desc  ndarray数据维度操作
"""
import numpy as np

# 维度操作分为两种:不会修改原始数据的维度-->1、视图变维（reshape）2、复制变维（ravel）:拉伸为1为
# 直接修改原始数据的维度：

a = np.arange(1, 9, 1)
print(a)
b = a.reshape(2, 4)
print(b)
c = a.reshape((2, 2, 2))
print(c)
d = c.ravel()
print(d)

e = c.flatten()  # 拉伸为1维
print(e)

# 就地变维
a.shape= (2,4)
print(a)
a.resize((2,2,2))
print(a)