#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  03_sklearn_lr_demo.py
:time  2023/12/7 10:07
:desc  sklearn线性回归
"""
from common_utils import *
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pickle

# 准备数据
data_path = f"{DATA_PATH}/Salary_Data.csv"
data = pd.read_csv(data_path)
# print(data.head())
# 构建模型
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 训练模型
model = lm.LinearRegression()
model.fit(x, y)
# 预测
pred_y = model.predict(x)
# 画图
plt.figure("Linear Regression")
plt.plot(x, pred_y, color="red")
plt.scatter(x, y)
plt.show()
# 评估
# 平均绝对误差
test_x = x.iloc[::4]
test_y = y.iloc[::4]
pred_test_y = model.predict(test_x)
print(sm.mean_absolute_error(test_y, pred_test_y))
# 均方误差
print(sm.mean_squared_error(test_y, pred_test_y))
# 中位数绝对偏差
print(sm.median_absolute_error(test_y, pred_test_y))

# r2-score
print(sm.r2_score(test_y, pred_test_y))

# 保存模型
model_path = "model.pickle"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
# 模型加载
with open(model_path, "rb") as f:
    model = pickle.load(f)
res = model.predict([[7.1]])
print(res)
