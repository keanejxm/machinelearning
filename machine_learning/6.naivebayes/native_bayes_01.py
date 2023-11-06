#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  native_bayes_01.py
:time  2023/11/6 9:33
:desc  朴素贝叶斯
"""
import numpy as np
import pandas as pd
import sklearn.naive_bayes as nb
import sklearn.model_selection as ms
import sklearn.metrics as sm
import matplotlib.pyplot as plt

data = pd.read_csv("multiple1.txt", names=["x1", "x2", "y"])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

train_x,test_x,train_y,test_y = ms.train_test_split(x,y,random_state=7,test_size=0.1,stratify=y)

model = nb.GaussianNB()

model.fit(train_x,train_y)

pred_y = model.predict(test_x)

print(sm.classification_report(test_y,pred_y))