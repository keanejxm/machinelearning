#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pd_demo_01.py
:time  2024/6/12 9:04
:desc  series(系列)的创建与属性
"""
import pandas as pd
import numpy as np

# 创建空的系列
a = pd.Series(dtype="float64")
print(a)

# 从ndarray创建一个Series
data = np.array(["张三", "李四", "王五", "赵六"])
s = pd.Series(data)
print(s)
s = pd.Series(data, index=["100", "101", "102", "103"])
print(s)
s = pd.Series(dtype="float32")  # data;index;dtype;name
print(s)
# 从字典创建一个Series
data = {'100': "张三", "101": "李四", "102": "王五", "103": "赵六"}
s = pd.Series(data)
print(s)
# 从标量创建一个Series
s = pd.Series(5, index=[0, 1, 2, 3])
print(s)

# 使用索引检索元素-->索引和切片
s = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
print(s[0], s[:3], s[-3:])
# 使用标签检索数据
print(s["a"], s["a":"c"], s[["a", "c", "d"]])

# --------------------------常用属性----------------------------------
print(s.values)
print(s.index)
print(s.dtype)
print(s.size)
print(s.ndim)
print(s.shape)
