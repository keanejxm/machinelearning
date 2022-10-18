#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  test_jxm.py
:time  2022/10/18 11:42
:desc  
"""
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])  # 2X3矩阵
y = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [0, 1, 1, 0]])  # 3X4矩阵
result = np.dot(x, y)  # 2X4矩阵
print(result)

a = np.array([[10,], [20,]])
b = np.array([[1, 2], [3, 4]])
print(a.dot(b))
