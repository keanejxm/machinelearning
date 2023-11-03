#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  decision_tree_1.py
:time  2023/11/2 15:07
:desc  决策树
"""
import pandas as pd

import sklearn.datasets as sd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.tree as st
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import matplotlib.pyplot as plt

boston_data = pd.read_csv("boston_house_prices.csv", header=1)
columns = boston_data.columns[:-1]
x = boston_data.iloc[:, :-1]
y = boston_data.iloc[:, -1]
# 划分训练集测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.2, random_state=7)


model = st.DecisionTreeRegressor(max_depth=10)
model.fit(train_x,train_y)

pred_test_y = model.predict(test_x)
pred_train_y = model.predict(train_x)

r2_score1 = sm.r2_score(pred_test_y,test_y)
r2_score2 = sm.r2_score(pred_train_y,train_y)
print(r2_score1,r2_score2)

fi = model.feature_importances_
fi = pd.Series(fi,index=columns)
fi = fi.sort_values(ascending=False)
print(fi)
st.plot_tree(model)
plt.show()