#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_regression_lasso_ridge.py
:time  2023/11/2 9:38
:desc  lasso回归与岭回归
"""
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_Data2.csv", header=0)

# 处理数据
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 线性回归
model = lm.LinearRegression()
model.fit(x, y)
# 岭回归
model_ridge = lm.Ridge(alpha=100)
model_ridge.fit(x, y)

# 预测
pred_y = model.predict(x)
pred_ridge_y = model_ridge.predict(x)
# 画图
plt.figure("Linear Regression")
plt.title("Linear Regression", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.scatter(x, y)
plt.plot(x, pred_y, c="blue")
plt.plot(x, pred_ridge_y, c="red")
plt.show()

# 模型评估
# 选择测试集
test_x = x.iloc[:30:4]
test_y = y.iloc[:30:4]
pred_test_y = model_ridge.predict(test_x)

r2_score_num = sm.r2_score(test_y, pred_test_y)
print(r2_score_num)
for i in range(90, 110, 1):
    # 训练模型
    model_ridge_1 = lm.Ridge(alpha=i)
    model_ridge_1.fit(x, y)

    # 测试那个模型得分搞
    pred_test_y_1 = model_ridge_1.predict(test_x)

    r2_score_num1 = sm.r2_score(test_y,pred_test_y_1)
    print(i,"----->",r2_score_num1)
