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
from sklearn.linear_model import LinearRegression


# import numpy as np
#
# train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
# train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集
# # 构建模型
# w = [1]
# b = [1]
# lr= 0.01
# # 更新w和b
# for i in range(1000):
#     pred_y = w * train_x + b
#     # 构建损失函数
#     # print((train_y-pred_y)**2)
#     # print(np.sqrt(train_y-pred_y))# 开根号
#     loss = np.mean(np.sum((train_y - pred_y)**2)) / 2
#
#     b_d = -(train_y-pred_y).sum()
#     w_d = -(train_x*(train_y-pred_y)).sum()
#
#     w= w-w_d*lr
#     b = b-b_d*lr
#     print(f"w:{w},b:{b},loss:{loss}")
#
# # 训练模型


class ManualLinear:
    def __init__(self):
        self.train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
        self.train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集
        self.w1 = [1]
        self.w2 = [1]
        self.lr = 0.01  # 学习率
        self.losses = list()
        self.epoches = list()

    def linear_regression(self):
        # y = kx + b
        # 损失函数：1/2（y-y`）**2
        # 梯度下降：初始值减去学习率*当前点切线方向移动的距离
        # f(x)=1/2(y-(kx+b))**2  对k求导:x(y`-y)
        #                        对b求导:(y`-y)

        for i in range(500):
            y_pred = self.w1[-1] * self.train_x + self.w2[-1]

            # 损失函数
            loss = 1 / 2 * ((y_pred - self.train_y) ** 2).sum()

            self.losses.append(loss)
            self.epoches.append(i)
            self.w1.append(self.w1[-1] - self.lr * (self.train_x * (y_pred - self.train_y)).sum())
            self.w2.append(self.w2[-1] - self.lr * (y_pred - self.train_y).sum())

            print(self.w1[-1], self.w2[-1], loss)

    def matplotlib_show(self):
        pred_y = self.w1[-1] * self.train_x + self.w2[-1]

        plt.figure(
            "Linear Regression",
            figsize=(4, 3),
            facecolor="lightgray"
        )
        plt.subplot(3, 1, 1)
        plt.scatter(self.train_x, self.train_y)
        plt.subplot(3, 1, 2)
        plt.plot(self.train_x, pred_y)
        plt.subplot(3, 1, 3)
        plt.plot( self.epoches, self.losses)
        plt.show()

    def start(self):
        self.linear_regression()
        self.matplotlib_show()


class SklearnLinearRegression:
    def __init__(self):
        self.train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
        self.train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集
        self.w1 = [1]
        self.w2 = [1]
        self.lr = 0.01  # 学习率
        self.losses = list()
        self.epoches = list()

    def linear_regression(self):
        model = LinearRegression()
        model.fit(self.train_x,self.train_y)




if __name__ == '__main__':
    linear_obj = ManualLinear()
    linear_obj.start()
