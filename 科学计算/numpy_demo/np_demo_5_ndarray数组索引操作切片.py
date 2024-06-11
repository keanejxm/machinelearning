#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_5_ndarray数组索引操作切片.py
:time  2024/6/11 15:18
:desc  数组的索引和切片操作
"""
import numpy as np

a = np.arange(1, 10)
print(a)
print(a[:3])
print(a[3:6])
print(a[6:])
print(a[::-1])
print(a[:-4:-1])
print(a[-7::-1])
print(a[::])
print(a[:])
print(a[::3])
print(a[1::3])
print(a[2::3])

a = np.arange(1, 28)
a.resize((3, 3, 3))
print(a)
print(a[1, :, :])  # 切出第一页
print(a[:,1,:])
print(a[0,:,1])

# 掩码操作
a = np.arange(1,10)
mask = [True,False,True,False,True,False,True,False,True]
print(a[mask])
