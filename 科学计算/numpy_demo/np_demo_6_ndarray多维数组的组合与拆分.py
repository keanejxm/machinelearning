#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_5_ndarray多维数组的组合与拆分.py
:time  2024/6/11 16:19
:desc  多维数组的组合与拆分
"""
import numpy as np

a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(2, 3)
print(a)
print(b)
# 垂直方向的组合
c = np.vstack((a, b))
print(c)
# 垂直方向的拆分
d = np.vsplit(c, 2)
print(d)

# ----------------------------水平方向的操作-----------------------------------
# 水平方向的组合
e = np.hstack((a, b))
print(e)
# 水平方向的拆分
f = np.hsplit(e, 2)
print(f)

# ----------------------------深度方向的操作-------------------------------------

# 深度方向3维完成组合操作
g = np.dstack((a, b))
print(g)
# 深度方向3维完成拆分操作---->只可拆分为3维
h = np.dsplit(g, 2)
print(h)

i = np.concatenate((a, b), axis=0)  # 0是垂直方向
print(i)
j = np.split(c,2)
print(j)
