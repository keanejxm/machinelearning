#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  decision_tree_boston.py
:time  2023/11/3 11:07
:desc  
"""
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.ensemble as se

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
model = se.RandomForestClassifier(max_depth=12, n_estimators=600, random_state=7,class_weight="balanced")
model.fit(train_x, train_y)

# 测试集
test_data = [
    ["high", "med", "5more", "4", "big", "low", "unacc"],
    ["high", "high", "4", "4", "med", "med", "acc"],
    ["low", "low", "2", "4", "small", "high", "good"],
    ["low", "low", "3", "4", "med", "high", "vgood"],
]

test_data = pd.DataFrame(test_data)
for i in test_data:
    encoder = encoders[i]
    res = encoder.transform(test_data[i])
    test_data[i] = res

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]
pred_y = model.predict(test_x)
print(pred_y)
print("真实类别：",encoders[6].inverse_transform(test_y.values))
print("测试类别：",encoders[6].inverse_transform(pred_y))
