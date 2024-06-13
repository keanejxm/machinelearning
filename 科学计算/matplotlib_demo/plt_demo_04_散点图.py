#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_04_散点图.py
:time  2024/6/12 17:23
:desc  散点图
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(172, 10, 20)
y = np.random.normal(60, 10, 20)

plt.scatter(
    x,
    y,
    s=10,  # 大小
    marker="D",  # 点型
    c="red",
)
plt.show()



