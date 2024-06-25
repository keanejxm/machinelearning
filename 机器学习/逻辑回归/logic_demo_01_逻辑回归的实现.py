#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 17:05
# @Author  : keane
# @Site    : 逻辑回归的简单实现
# @File    : logic_demo_01_逻辑回归的实现.py
# @Software: PyCharm
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self):
        self.x = np.array([[3, 1], [2, 5], [1, 8], [6, 4], [5, 2], [3, 5], [4, 7], [4, -1]])
        self.y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

    def build_model(self):
        """
        构建模型
        Returns:

        """
        model = lm.LogisticRegression(solver='liblinear',C=1)
        model.fit(self.x, self.y)
        pred_y = model.predict(self.x)
        print(pred_y)

    def build_logistic_function(self):
        """
        构建逻辑函数
        Returns:

        """
        x = np.linspace(-10,10,100)
        y = 1/(1+np.e**-x)
        plt.plot(x,y)
        plt.show()


if __name__ == '__main__':
    logistic_reg = LogisticRegression()
    logistic_reg.build_model()
    logistic_reg.build_logistic_function()
