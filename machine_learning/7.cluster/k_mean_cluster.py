#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_cluster.py
:time  2023/11/6 14:30
:desc  
"""

import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt

data = pd.read_csv("multiple3.txt", header=None, names=["x1", "x2"])
# print(data.head())
model = sc.KMeans(n_clusters=4)

model.fit(data)

label = model.labels_
print(label)

center = model.cluster_centers_
print(center)
plt.scatter(data["x1"], data["x2"], c=label, cmap="brg")
plt.colorbar()
# 集合中心
plt.scatter(center[:, 0], center[:, 1], marker="+", s=300, color="black")
plt.show()
