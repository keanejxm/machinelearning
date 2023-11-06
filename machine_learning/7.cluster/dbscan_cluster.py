#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  dbscan_cluster.py
:time  2023/11/6 14:49
:desc  DBSCAN算法
"""

import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt

data = pd.read_csv("multiple3.txt",header= None,names=["x1","x2"])

model = sc.DBSCAN(eps=0.65,min_samples=5)

model.fit(data)

label = model.labels_
# 轮廓系数

print(label)

plt.scatter(data["x1"],data["x2"],c = label,cmap="brg")

plt.colorbar()
plt.show()