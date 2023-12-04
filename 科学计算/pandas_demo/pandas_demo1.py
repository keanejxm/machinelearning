#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pandas.py
:time  2023/11/23 10:49
:desc  pandas
"""
"""
1、pandas核心数据结构
    Series:一维数组
    DataFrame:二维数组
2、Series:
    创建：列表创建、字典创建、标量
    检索：使用索引检索、使用标签检索、索引;切片;掩码
        Series中的位置索引没有反向索引，设置索引后可以使用反向索引
    属性：values、index、dtype、size、ndim、shape
3、DataFrame:
    特点：
        列和列之间可以是不同类型
        大小可变
        标记轴
        针对行和列进行轴向统计
    创建：列表、二维列表、字典、
    属性：axes、columns、index、dtypes、empty、ndim、size、values、head(n)、tail(n)
4、核心数据的操作：
    列访问：索引、访问一列、访问多列
        列级索引没有位置索引，只有标签索引
    列添加：列表、Series(指定index)
    列删除：del、pop、drop
    
    行访问：能够直接切片，不能直接索引
        loc:操作标签索引
        iloc:操作位置索引
    行添加：append
    行删除：drop
    行的修改：
        通过列找行可以修改
        通过行找列不能修改
"""
import pandas as pd
import numpy as np

# Series

ser = pd.Series([1, 2, 3, 4])
print(ser)
ser = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
ser = pd.Series(np.array([1, 2, 3, 4]), index=["a", "b", "c", "d"])
data = {"aa": 100, "bb": 200, "cc": 300}
ser = pd.Series(data)
print(ser["aa"])
print(ser.iloc[0])
print(ser.iloc[:2])
# 属性
print(ser.values)
print(ser.dtype)
print(ser.ndim)
print(ser.shape)
print(ser.index)
print(ser.size)

# dataframe
# 1、创建
df = pd.DataFrame([1, 2, 3, 4])
print(df)
df = pd.DataFrame([[1, 2, 3, 4],
                   [5, 6, 7, 8]])
print(df)
df = pd.DataFrame({"aa": [1, 2, 3, 4],
                   "bb": [5, 6, 7, 8],
                   "cc": [10, 11, 12, 13]})
print(df)
df = pd.DataFrame({"aa": pd.Series([1, 2, 3, 4]),
                   "bb": pd.Series([5, 6, 7, 8]),
                   "cc": pd.Series([1, 2, 34])})
print(df)
# 2、属性
print(df.axes)
print(df.columns)
print(df.index)
print(df.dtypes)
print(df.empty)
print(df.ndim)
print(df.size)
print(df.values)
print(df.head(1))
print(df.tail(1))
# 核心数据操作
# 列操作
data = {
    "one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
    "two": pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"]),
    "three": pd.Series([1, 2, 3], index=["a", "b", "c"]),
}
df = pd.DataFrame(data)
print(df)
print(df["one"])
print(df.columns[:2])
# 列添加
df["four"] = [4, 5, 6, 7]
df["five"] = pd.Series([2, 3, 4, 5], index=["a", "b", "c", "d"])
print(df)
# 列删除
del df["five"]
df.pop('four')
df.drop(["one", "two"], axis=1, inplace=True)
print(df)

###行操作
# 行访问

data = {
    "one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
    "two": pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"]),
    "three": pd.Series([1, 2, 3], index=["a", "b", "c"]),
}
df = pd.DataFrame(data)
print(df)

###行操作
# 行访问
# loc
row = df.loc["a":"b"]  # 切片
row = df.loc["a", "one":"two"]  # 索引切片
row = df.loc["a"]  # 索引
row = df.loc[:, "one":"two"]
row = df.loc[["a", "c"]]  # 掩码操作
# iloc
row = df.iloc[0]  # 索引
row = df.iloc[0, :-1]  # 索引切片
row = df.iloc[0:1, -1]  # 切片索引
row = df.iloc[0:2, 0:2]  # 切片切片
row = df.iloc[[0, 2]]  # 掩码操作

print(row)
# 行添加