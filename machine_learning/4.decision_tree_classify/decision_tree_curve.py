#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  decision_tree_boston.py
:time  2023/11/3 11:07
:desc  
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

# 了解数据特征
# 读取数据
data = pd.read_csv("car.txt", header=None)
train_data = pd.DataFrame()
encoders = dict()
for i in data:
    encoder = sp.LabelEncoder()
    res = encoder.fit_transform(data[i])
    train_data[i] = res
    encoders[i] = encoder
# 整理输入输出
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]

# 构架模型
model = se.RandomForestClassifier(max_depth=6, n_estimators=400, random_state=7)

params = np.arange(100, 1001, 100)

train_score, test_score = ms.validation_curve(model,
                                              train_x, train_y,
                                              param_name='n_estimators',
                                              param_range=params,
                                              cv=5
                                              )
average_core = test_score.mean(axis = 1)
plt.plot(params,average_core,"o-")
plt.show()