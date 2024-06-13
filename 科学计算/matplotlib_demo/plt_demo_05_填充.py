#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_05_填充.py
:time  2024/6/12 17:29
:desc  填充
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 8 * np.pi, 1000)
sin_y = np.sin(x)
cos_y = np.cos(x)
plt.figure('Fill', facecolor="lightgray")
plt.title("Fill", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.tick_params(labelsize=10)
plt.grid(linestyle=":")
plt.plot(x, sin_y, c="dodgerblue", label=r"$y=sin(x)$")
plt.plot(x, cos_y, c="orangered", label=r"$y=\frac{1}{2}cos(\frac{x}{2})$")

# 填充cos_y<sin_y
plt.fill_between(x, cos_y, sin_y, cos_y < sin_y, color='dodgerblue', alpha=0.5)

# 填充cos_y>sin_y
plt.fill_between(x, cos_y, sin_y, cos_y > sin_y, color="orangered", alpha=0.5)

plt.legend()
plt.show()
