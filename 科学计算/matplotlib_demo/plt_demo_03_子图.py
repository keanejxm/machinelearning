#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_03_子图.py
:time  2024/6/12 17:06
:desc  子图
"""
import matplotlib.pyplot as plt

plt.figure(
    "子图",
    figsize=(12, 9),
    facecolor="lightgray"
)
# 绘制9宫格矩阵式子图，每个子图写一个数字
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.text(0.5,
             0.5,
             s=i + 1,
             ha="center",
             va="center",
             size=36,
             alpha=0.5,
             # withdash=False,
             )
plt.legend()  # 图例
plt.tight_layout()  # 紧凑模式
# plt.grid(linestyle="--")  # 网格
plt.show()

# 自由布局
plt.figure(
    "自由布局",
    figsize=(10, 5),
    facecolor="lightgray"
)
plt.title("自由布局")
plt.axes([0.03, 0.03, 0.94, 0.94])
plt.text(0.5,0.5,1,ha="center",va="center",size=36)
plt.show()
