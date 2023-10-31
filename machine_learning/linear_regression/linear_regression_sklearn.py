#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_regression_sklearn.py
:time  2023/10/31 15:59
:desc  线性回归sklearn实现
"""
import numpy as np
import sklearn.linear_model as lm  # 线性模块-线性模型
import sklearn.metrics as sm  # 模型性能评价模块
import matplotlib.pyplot as plt

train_x = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]])  # 输入集
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集

# 创建线性回归器
model = lm.LinearRegression()

# 用已知输入、输出数据集训练回归器
model.fit(train_x, train_y)

# 根据训练模型预测输出
pred_y = model.predict(train_x)

print("coef_:", model.coef_)  # 系数
print("intercept_:", model.intercept_)  # 截距

# 可视化回归曲线
plt.figure("Linear Regression", facecolor="lightgray")
plt.title("Linear Regression", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.grid()
plt.tight_layout()
# 绘制样本点
plt.scatter(train_x, train_y, label="training", c='blue', alpha=0.8, s=60)
# 绘制拟合曲线
plt.plot(train_x, pred_y, label="regression",c ='orangered')
plt.legend()
plt.show()
