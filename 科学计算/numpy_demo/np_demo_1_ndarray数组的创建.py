#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_1.py
:time  2024/6/11 10:25
:desc  ndarray数组的创建
"""

import numpy as np

arr1 = np.array([1,2,3,4,5])
# nd数组与列表表现形式不同之处，在于nd数组没有逗号，而列表有逗号
print(arr1)
arr2 = np.ones((3,3))
print(arr2)

arr3 = np.zeros((3,3))
print(arr3)

arr4 = np.arange(1,10,2)
print(arr4)