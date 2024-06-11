#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  np_demo_3_ndarray数组的基本数据类型.py
:time  2024/6/11 11:00
:desc  数组的基本数据类型
"""
import numpy as np

# bool_:布尔
arr = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
print(arr.dtype)
arr1 = arr.astype(np.bool_)
print(arr1)
print(arr1.dtype)
# int8:有符号整型-->会带有正负号
arr2 = arr.astype(np.int8)
print(arr2)
print(arr2.dtype)
# uint8:无符号整型
arr3 = arr.astype(np.uint8)
print(arr3)
print(arr3.dtype)
# float32:浮点型
arr4 = arr.astype(np.float32)
print(arr4)
print(arr4.dtype)
# 字符串类型
arr5 = arr.astype(np.str_)
print(arr5)
print(arr5.dtype)
# 时间类型
arr6 = np.array(["2011", "2012-01-01", "2023-01-01 01:01:01", "2011-02-01"])
arr7 = arr6.astype(np.datetime64)
print(arr7)
print(arr7.dtype)
arr8 = arr6.astype("M8[D]")
print("arr8:",arr8)
print(arr8.dtype)
arr9 = arr6.astype("M8[Y]")
print("arr9:",arr9)
print(arr9.dtype)
arr10 = arr6.astype("M8[M]")
print("arr10:",arr10)
print(arr10.dtype)
# 自定义复合类型
# 第一种
data = [
    ('zs', [90, 80, 85], 15),
    ('ls', [92, 81, 83], 16),
    ('ww', [95, 85, 95], 15)
]
a = np.array(data, dtype='U3,3int,int32')
print(a)
print(a[0]['f0'], ":", a[1]['f1'])

# 第二种
c = np.array(data, dtype={"names": ["name", "scores", "ages"],
                          "formats": ["U3", "3int32", "int32"]})
print(c)
print(c[0]["name"],":",c[0]["scores"],":",c.itemsize)
