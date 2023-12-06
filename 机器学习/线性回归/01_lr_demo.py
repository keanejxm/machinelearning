#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_lr_demo.py
:time  2023/12/6 17:02
:desc  手动实现线性回归
"""
import numpy as np

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集
# 构建模型
w = [1]
b = [1]
lr= 0.01
# 更新w和b
for i in range(1000):
    pred_y = w * train_x + b
    # 构建损失函数
    # print((train_y-pred_y)**2)
    # print(np.sqrt(train_y-pred_y))# 开根号
    loss = np.mean(np.sum((train_y - pred_y)**2)) / 2

    b_d = -(train_y-pred_y).sum()
    w_d = -(train_x*(train_y-pred_y)).sum()

    w= w-w_d*lr
    b = b-b_d*lr
    print(f"w:{w},b:{b},loss:{loss}")

# 训练模型