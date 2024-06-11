#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_2_ndarray数组的属性.py
:time  2024/6/11 10:29
:desc  ndarray数组的属性
"""
import numpy as np

# -------------------------shape--------------------
# arr = np.ones((3, 3))
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.shape)  # 维度
arr.shape = (2, 3)  # 转为二维数组
print(arr)
arr.shape = (6,)
print(arr)
arr.shape = (1, 2, 3)  # 转为三维数组
print(arr)
arr.shape = (6,)
# -------------------------dtype--------------------
print(arr.dtype)  # 查看元素类型：int32

arr1 = arr.astype(np.float32)  # 转为float32类型，转化后原数据不变
print(arr1.dtype)

# -------------------------size--------------------
arr.shape = (3, 2)
print(arr.size)  # 数组的个数
print(len(arr))  # 数组的长度（数组的行数）
# size和len的区别

# -------------------------索引下标--------------------

arr = np.arange(1,9,1)

arr.shape = (2,2,2)
print(arr)
print(arr[0])
print(arr[0][0])
print(arr[0][0][0])
print(arr[0,0,0])
