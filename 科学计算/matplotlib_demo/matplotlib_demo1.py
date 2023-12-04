#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  matplotlib_demo1.py
:time  2023/12/1 10:12
:desc  
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 5, 7, 9, 13])
y = np.array([11, 5, 13, 4, 21])

plt.plot(x, y)
plt.show()

# sinx
x = np.linspace(-np.pi, np.pi, 200)
y = np.sin(x)
plt.plot(x, y,)
plt.show()
