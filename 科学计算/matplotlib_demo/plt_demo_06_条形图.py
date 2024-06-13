#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_06_条形图.py
:time  2024/6/12 17:42
:desc  
"""
import matplotlib.pyplot as plt
import numpy as np

apples = np.array([30, 25, 22, 36, 21, 29, 20, 24, 33, 19, 27, 15])
oranges = np.array([24, 33, 19, 27, 35, 20, 15, 27, 20, 32, 20, 22])
plt.figure("bar", facecolor="lightgray")
plt.title("bar")
plt.xlabel("month", fontsize=15)
plt.ylabel("price", fontsize=15)
plt.tick_params(labelsize=10)
plt.grid(axis="y", linestyle=":")
plt.ylim((0, 40))
x = np.arange(len(apples))
plt.bar(
    x - 0.2,
    apples,
    0.4,
    color = "dodgerblue",
    label ="Apple"
)
plt.bar(
    x+0.2,
    oranges,
    0.4,
    color="orangered",
    label="Orange",
    alpha=0.75
        )
# plt.xticks()
plt.legend()
plt.show()