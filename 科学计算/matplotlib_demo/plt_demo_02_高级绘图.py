#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  plt_demo_02_高级绘图.py
:time  2024/6/12 16:39
:desc  
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建窗口
plt.figure(
    "title",  # 标题
    figsize=(4, 3),  # 窗口大小
    facecolor="lightgray"  # 图表背景色
)
# 设置标题
plt.title("11111")
# x轴，y轴的名字
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
# 设置刻度参数

# 网格先
plt.grid(linestyle="--")
# 紧凑布局
plt.tight_layout()

plt.show()
