#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  06_sklearn_boston_demo.py
:time  2023/12/8 14:33
:desc  
"""
import sklearn.datasets as sd
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.metrics as sm

boston = sd.load_boston()

# print(boston.keys())
# print(boston.filename)
# print(boston.DESCR)
# print(boston.feature_names)
# print(boston.data)
# print(boston.target)

x = boston.data
y = boston.target
# 划分训练集测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1,shuffle=True,random_state=7)
# print(train_x, train_y)

# 构建模型
model = pl.make_pipeline(sp.PolynomialFeatures(2),
                         lm.LinearRegression())

model.fit(train_x,train_y)

pred_train_y = model.predict(train_x)

# 构建损失函数（回归）


pred_y = model.predict(test_x)




r2_score = sm.r2_score(test_y,pred_y)
print(r2_score)
