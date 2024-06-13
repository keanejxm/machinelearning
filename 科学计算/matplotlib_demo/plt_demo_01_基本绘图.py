#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_01.py
:time  2024/6/12 15:34
:desc  基本绘图
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# 绘图
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])

plt.plot(x, y)
plt.show()

# 设置线型、线宽、颜色、透明度--->绘制正弦、余弦、并设置线型、线宽、颜色、透明度

# x = np.arange(-np.pi, np.pi, 1)
x = np.linspace(-np.pi, np.pi, 100)
sin_y = np.sin(x)
cos_y = np.cos(x)

plt.plot(x, sin_y, label="sin", linestyle="--", linewidth=4, color="red", alpha=0.5)
plt.plot(x, cos_y, label="cos", linestyle="-.", linewidth=3, color="blue", alpha=0.6)

plt.xlabel("x")
plt.ylabel("y")

# 设置坐标轴范围
plt.xlim(-math.pi, 2 * math.pi)
plt.ylim(-1, 2)

plt.title("sin&cos")
plt.legend()  # 图例

# 设置坐标轴刻度
x_tck = np.arange(-np.pi, np.pi)
x_txt = np.array([r"$-\frac{\pi}{2}$", "-2", "-1", "0", "1", "2", r"$\frac{\pi}{2}$"])
y_tck = np.arange(-2, 2, 1)
y_txt = y_tck.astype("U")
plt.xticks(x_tck, x_txt)
plt.yticks(y_tck, y_txt)

# 设置坐标轴
ax = plt.gca()
axis_b = ax.spines["bottom"]
axis_b.set_position(('data', 0))
axis_l = ax.spines["left"]
axis_l.set_position(("data", 0))
# 顶部设置无色，右侧设置无色
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
# 特殊点
plt.scatter(x_tck, np.sin(x_tck),
            marker="s",  # 点形状
            s=40,  # 点大小
            facecolor="blue",  # 填充色
            zorder=3  # 绘制图层编号
            )

plt.show()
