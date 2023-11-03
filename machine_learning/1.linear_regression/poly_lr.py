#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  poly_lr.py
:time  2023/11/2 10:09
:desc  多项式回归
"""
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import matplotlib.pyplot as plt
data = pd.read_csv("Salary_Data.csv")

# 整理数据
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 构建模型
model = pl.make_pipeline(sp.PolynomialFeatures(3),lm.LinearRegression())

# 训练

model.fit(x,y)
pred_y = model.predict(x)
plt.plot(x,pred_y,color = "orangered")
plt.scatter(x,y)
plt.show()

# 画图
