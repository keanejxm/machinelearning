#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  agg_cluster.py
:time  2023/11/6 15:06
:desc  
"""
import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt

data = pd.read_csv("multiple3.txt", header=None, names=["x1", "x2"])

model = sc.AgglomerativeClustering(n_clusters=4)

model.fit(data)

label = model.labels_
print(label)

plt.scatter(data["x1"], data["x2"], c=label, cmap="brg")

plt.colorbar()
plt.show()
