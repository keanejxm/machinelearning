#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_07_直方图.py
:time  2024/6/12 17:51
:desc  
"""
import numpy as np
import matplotlib.pyplot as plt

mu = 100
sigma = 15
x = mu + sigma * np.random.randn(200)
num_bins = 25
plt.figure("直方图")
n, bins, patches = plt.hist(x, num_bins, color="red")
plt.show()
