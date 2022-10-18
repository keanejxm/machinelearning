#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_3.py
:time  2022/10/18 15:22
:desc  联系numpy:科学计算包和pandas
"""
import numpy as np
import pandas as pd

l = [1, 2, 3, 4]
nd_l = np.array(l, dtype=np.float32, ndmin=1)
print(nd_l)
nd_zero = np.zeros((3,4))
print(nd_zero)
nd_empty = np.empty((2,3))
print(nd_empty)
nd_ones = np.ones((2,3))
print(nd_ones)
print(nd_ones.shape)
print(nd_ones.size)
print(nd_ones.dtype)
print(nd_ones.ndim)
print(nd_ones.T)


nd_data = np.array([[1,2,3,4],[5,6,7,8]])
print(nd_data.sum())
print(nd_data.sum(axis=0))
print(nd_data.sum(axis=1))
print(nd_data.max())
print(nd_data.max(axis=0))
print(nd_data.max(axis=1))
print(nd_data.min())
print(nd_data.min(axis=0))
print(nd_data.min(axis=1))
print(nd_data.mean())
print(nd_data.mean(axis=0))
print(nd_data.mean(axis=1))
print(nd_data.var())
print(nd_data.var(axis=0))
print(nd_data.var(axis=1))
