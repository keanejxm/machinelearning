#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pd_demo_03_DataFrame的核心数据结构操作.py
:time  2024/6/12 10:28
:desc  DataFrame的核心数据结构操作
"""
import pandas as pd

# -------------------------------列访问、添加、删除-----------------------------------
# 列访问
data = {"one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
        "two": pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"]),
        "three": pd.Series([1, 3, 4], index=["a", "c", "d"])}
df = pd.DataFrame(data)
print(df)
print(df["one"])  # 索引降维
print(df[df.columns[:2]])  # 切片不降维

# 列添加
df["four"] = pd.Series([90, 80, 70, 60], index=["a", "b", "c", "d"])
print(df)

# 列删除
del (df["one"])
print(df)
column_two = df.pop("two")
print(df)
print(column_two)

# 列删除多列
df2 = df.drop(["three", "four"], axis=1, inplace=True)
print(df2)

# ---------------------------------行的访问、添加和删除--------------------------------
d = {"one": pd.Series([1, 2, 3], index=["a", "b", "c"]),
     "two": pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])}
df = pd.DataFrame(d)
# 切片
print(df[2:4])

# loc切片，只支持名称索引，不支持位置索引
print(df.loc["a"])
print(df.loc["a":"b"])
print(df.loc[["a", "b"]])

# iloc 位置索引
print(df.iloc[2])
print(df.iloc[[2, 3]])

# 行添加
df1 = pd.DataFrame([['zs', 12], ['ls', 4]], columns=['Name', 'Age'])
df2 = pd.DataFrame([['ww', 16], ['zl', 8]], columns=['Name', 'Age'])
df3 = df1.append(df2)
print(df3)
# 删除index为0的行
# df3 = df3.reset_index()# 重置索引
df4 = df3.drop(0)
print(df4)

df3['Name'][0] = 'Tom'
print(df3)
