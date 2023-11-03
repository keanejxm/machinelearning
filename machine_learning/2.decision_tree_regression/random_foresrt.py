#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  random_foresrt.py
:time  2023/11/2 17:03
:desc  随机森林
"""
#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.tree as st
import sklearn.ensemble as se

boston_df = pd.read_csv("boston_house_prices.csv", header=1)
columns = boston_df.columns[:-1]
x = boston_df.iloc[:, :-1]
y = boston_df.iloc[:, -1]
# 划分训练集测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7)


model = se.RandomForestRegressor(max_depth=6,n_estimators=400,random_state=7)

# 训练
model.fit(train_x, train_y)

# 预测
pred_train_y = model.predict(train_x)
pred_test_y = model.predict(test_x)

r2_score_train = sm.r2_score(train_y, pred_train_y)
r2_score_test = sm.r2_score(test_y, pred_test_y)

print(r2_score_train, r2_score_test)
