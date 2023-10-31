#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plot_lr.py
:time  2023/10/31 16:34
:desc  多项式线性回归
"""
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import sklearn.preprocessing as sp  # 数据预处理
# import sklearn.linear_model as lm
# import sklearn.metrics as sm
# import sklearn.pipeline as pl
# import matplotlib.pyplot as mp
#
# df = pd.read_csv("Salary_Data.csv")
#
# # 数据
# x = df.iloc[0:, :-1]
# y = df.iloc[0:, -1]
#
# # model = lm.LinearRegression()
# model = pl.make_pipeline(sp.PolynomialFeatures(10), lm.LinearRegression())
# # 训练模型
# model.fit(x, y)
#
# # 预测
# pred_y = model.predict(x)
# # 画图
# plt.figure("plot", facecolor="lightgray")
# plt.title("plot", fontsize=20)
# plt.xlabel("x", fontsize=14)
# plt.ylabel("y", fontsize=14)
# plt.grid()
# plt.scatter(x, y, c="red", alpha=0.8, s=60, label='sample')
# plt.plot(x, pred_y, c="blue", label="regression")
# plt.legend()
# plt.show()
'''
多项式回归
'''
import pandas as pd
import sklearn.preprocessing as sp #数据预处理
import sklearn.linear_model as lm #线性模型
import sklearn.pipeline as pl #管线模块
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')

#整理输入和输出
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

#构建模型
model = pl.make_pipeline(sp.PolynomialFeatures(3),
                         lm.LinearRegression())


model.fit(x,y)
pred_y = model.predict(x)
plt.plot(x,pred_y,color='orangered')
plt.scatter(x,y)
plt.show()