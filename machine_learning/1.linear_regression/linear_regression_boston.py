#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_regression_boston.py
:time  2023/11/2 10:34
:desc  波士顿房价预测
"""
import pandas as pd

import sklearn.datasets as sd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.metrics as sm

boston_data = pd.read_csv("boston_house_prices.csv", header=1)
x = boston_data.iloc[:, :-1]
y = boston_data.iloc[:, -1]
# 划分训练集测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.2, random_state=7)

for i in range(1,10,1):
    # 根据训练集训练模型
    # 线性回归
    model_lr = pl.make_pipeline(sp.PolynomialFeatures(i), lm.LinearRegression())
    model_lr.fit(train_x, train_y)

    # 岭回归
    model_ridge = pl.make_pipeline(sp.PolynomialFeatures(i), lm.Ridge())
    model_ridge.fit(train_x, train_y)

    # lasso回归
    model_lasso = pl.make_pipeline(sp.PolynomialFeatures(i), lm.Lasso())
    model_lasso.fit(train_x, train_y)

    for j,model in enumerate([model_lr, model_lasso, model_ridge]):
        # 根据测试集测试模型
        pred_test_y = model.predict(test_x)
        pred_train_y = model.predict(train_x)
        # 评估
        # 评估训练得分
        train_r2 = sm.r2_score(train_y, pred_train_y)
        # 评估测试得分
        test_r2 = sm.r2_score(test_y, pred_test_y)
        if j ==0:
            model_name = "线性回归"
        elif j==1:
            model_name = "lasso回归"
        else:
            model_name = "岭回归"
        print(model_name,i,train_r2,test_r2,f"差值{abs(train_r2-test_r2)}")


    # 如果训练得分>测试得分欠拟合，如果训练得分<测试得过拟合
