#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  em_cluster.py
:time  2023/11/6 17:56
:desc  
"""
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt

data = pd.read_csv("multiple3.txt", header=None, names=["x1", "x2"])
# print(data.head())
model = mixture.GaussianMixture(n_components=4)

model.fit(data)

label = model.predict(data)
print(label)
plt.scatter(data["x1"], data["x2"], c=label, cmap="brg")
plt.colorbar()
# 集合中心
plt.show()