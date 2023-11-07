#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pca_01.py
:time  2023/11/7 15:34
:desc  
"""
import sklearn.datasets as sd
from sklearn.decomposition import PCA
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as ms

iris = sd.load_iris()

x = iris.data
y = iris.target

# 移除均值
train_x = sp.StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

new_x = pca.fit_transform(train_x)

print(new_x)

data_x = pd.DataFrame(new_x, columns=["x1", "x2"])
plt.scatter(data_x["x1"], data_x["x2"], c=y, cmap="brg")
plt.show()

train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.2, random_state=7, stratify=y)

model = lm.LogisticRegression(solver="liblinear")
model.fit(train_x, train_y)

pred_y = model.predict(test_x)

print(sm.classification_report(test_y, pred_y))
