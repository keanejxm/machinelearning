#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_08_x为二维数据时画图.py
:time  2024/6/14 11:03
:desc  
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([[0.5,0.4], [0.6,0.6], [0.8,0.8], [1.1,1.2], [1.4,1.6]])
y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

plt.scatter(x,y)
plt.show()