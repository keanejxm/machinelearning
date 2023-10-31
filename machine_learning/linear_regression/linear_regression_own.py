#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_demo.py
:time  2023/10/31 14:28
:desc  线性回归自己代码实现
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sklearn.preprocessing as sp

# 训练数据集
train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集

n_epochs = 1000  # 迭代次数
l_rate = 0.01  # 学习率
epochs = []  # 记录迭代次数
losses = []  # 记录损失值
w0, w1 = [1], [1]  # 模型初始值

for i in range(1, n_epochs + 1):
    epochs.append(i)
    # 计算公式(一元一次函数)
    y = w0[-1] + w1[-1] * train_x
    # 损失函数
    loss = (((train_y - y) ** 2).sum()) / 2
    losses.append(loss)  # 记录每次迭代的损失值

    print(f"w0={w0[-1]};w1={w1[-1]};loss = {loss}")

    # 计算w0，w1的偏导数
    d0 = -(train_y - y).sum()
    d1 = -(train_x * (train_y - y)).sum()

    # 更新w0,w1
    w0.append(w0[-1] - (d0 * l_rate))
    w1.append(w1[-1] - (d1 * l_rate))

# 可视化
w0 = np.array(w0[:-1])
w1 = np.array(w1[:-1])

# 次数与损失函数曲线
plt.figure("Losses", facecolor='lightgray')
plt.title("epoch", fontsize=20)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.grid(linestyle=":")  # 网格线
plt.plot(epochs, losses, c="blue", label="loss")
plt.legend()  # 图例
plt.tight_layout()  # 紧凑格式

# 模型曲线
pred_y = w0[-1] + train_x * w1[-1]
plt.figure("Linear Regression", facecolor="lightgray")
plt.title("Linear Regression", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.grid(linestyle="")
plt.scatter(train_x, train_y, c="blue", label="training")
plt.plot(train_x, pred_y, c="red", label="Regression")
plt.legend()# 图例


# 展示梯度下降过程

plt.show()
