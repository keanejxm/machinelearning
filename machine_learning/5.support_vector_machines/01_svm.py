#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_svm.py
:time  2023/11/3 16:32
:desc  
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

data = pd.read_csv("multiple2.txt", header=None, names=["x1", "x2", "y"])
print(data.head())

plt.scatter(data["x1"], data["x2"], c=data["y"], cmap="brg")
plt.colorbar()
# plt.show()
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7, stratify=y)

# model = svm.SVC(kernel="linear")
# model = svm.SVC(kernel="poly",degree=3)
model = svm.SVC(kernel="rbf",gamma=0.1,C=1)
model.fit(train_x, train_y)

pred_test_y = model.predict(test_x)

# print(model.coef_)
# print(model.intercept_)

print(sm.classification_report(test_y, pred_test_y))

x1_min = data["x1"].min()
x1_max = data["x1"].max()

x2_min = data["x2"].min()
x2_max = data["x2"].max()

data_x = list()
for i in np.linspace(x1_min, x1_max, 200):
    for j in np.linspace(x2_min, x2_max, 200):
        data_x.append([i, j])

data_x = pd.DataFrame(data_x, columns=["x1", "x2"])

pred_y = model.predict(data_x)

plt.scatter(data_x["x1"], data_x["x2"], c=pred_y, cmap="gray")
plt.scatter(data["x1"], data["x2"], c=data["y"], cmap="brg")
plt.show()
