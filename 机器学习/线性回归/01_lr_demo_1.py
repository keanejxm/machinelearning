#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_lr_demo.py
:time  2023/12/6 17:02
:desc  手动实现线性回归
"""
import numpy as np
import matplotlib.pyplot as plt

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集

# y = kx + b
w1 = [1]
w2 = [1]
lr = 0.001  # 学习率
# f(x)=1/2(y-(kx+b))**2  对k求导:x(y`-y)
#                        对b求导:(y`-y)

for i in range(100):
    y_pred = w1[-1] * train_x + w2[-1]

    w1.append(w1[-1] + lr * (train_x * (y_pred - train_y)).sum())
    w2.append(w2[-1] + (y_pred - train_y).sum())

    print(w1[-1],w2[-1])








