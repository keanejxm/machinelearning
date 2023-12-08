#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  05_sklearn_poly_demo.py
:time  2023/12/8 14:10
:desc  多项式回归
"""
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import sklearn.pipeline as pl
from common_utils import *
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(f"{DATA_PATH}/Salary_Data.csv")

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 构建模型
model = pl.make_pipeline(sp.PolynomialFeatures(5),
                         lm.LinearRegression())

model.fit(x, y)
pred = model.predict(x)

plt.plot(x, pred, color="orangered")
plt.scatter(x, y)
plt.show()
