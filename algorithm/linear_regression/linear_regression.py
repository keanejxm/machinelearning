#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_regression.py
:time  2023/3/1 10:15
:desc  线性回归可视化
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 一元线性回归 y = ax+b  (y^ - ax+b)的平方  y真实值，ax+b 预测值  求真实值与预测值的误差和的平方

x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)
w, b = np.meshgrid(x, y)
a = 2-w-b
SSE = (2 - w - b) ** 2 + (4 - 3 * w - b) ** 2
ax = plt.axes(projection='3d')
ax.plot_surface(w, b, SSE, cmap='rainbow')
#生成z方向投影，投到x-y平面
ax.contour(w, b, SSE, zdir='z', offset=0, cmap="rainbow")
#x轴标题
plt.xlabel('w')
#y轴标题
plt.ylabel('b')
plt.show()

