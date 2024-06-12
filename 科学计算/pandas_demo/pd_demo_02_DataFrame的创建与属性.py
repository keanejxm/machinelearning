#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pd_demo_02_DataFrame的创建与属性.py
:time  2024/6/12 9:46
:desc  DataFrame的创建与属性
"""
import pandas as pd

# 创建一个DataFrame
a = pd.DataFrame()  # data;index;columns;dtype
print(a)
# 从列表创建DataFrame
data = [1, 2, 3, 4, 5]
b = pd.DataFrame(data)
print(b)
data = [["Alex", 10], ["Bob", 12], ["Clarke", 13]]
c = pd.DataFrame(data, columns=["name", "age"])
print(c)
d = pd.DataFrame(data, columns=["name", "age"], dtype=float)
print(d)
data = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
e = pd.DataFrame(data)
print(e)

# 从字典创建DataFrame
data = {"name": ["Tom", "jack", "Steve", "Ricky"], "Age": [28, 34, 29, 42]}
f = pd.DataFrame(data, index=["s1", "s2", "s3", "s4"])
print(f)

data = {"one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
        "two": pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])}

g = pd.DataFrame(data)  # 使用Series,当列表长度不一样是对应没有数据的索引为Nan
print(g)

# ------------------------------DataFrame属性--------------------------
print(g.axes)  # 返回行列标签
print(g.columns)  # 返回列标签
print(g.index)  # 返回行标签
print(g.dtypes)  # 返回对象的数据类型
print(g.empty)  # 如果DataFrame为空返回True
print(g.ndim)  # 返回底层数据的维度
print(g.size)  # 返回基础数据中的元素数
print(g.values)  # 将系列作为ndarray返回
print(g.head(2))  # 返回DataFrame的前n行
print(g.tail(2))  # 返回DataFrame的后n行
