#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  single_layer_perceptron.py
:time  2022/11/3 9:51
:desc  单层感知机
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

X = np.array(
    [
        [1, 4, 3],
        [1, 5, 4],
        [1, 4, 5],
        [1, 1, 1],
        [1, 2, 1],
        [1, 3, 2]
    ]
)
Y = np.array([1, 1, 1, -1, -1, -1])

W = (np.random.random(3) - 0.5) * 2
lr = 0.3
n = 0
O = 0


def update():
    global X, Y, W, lr, n
    n = n + 1
    print(np.dot(X,W.T))
    O = np.sign(np.dot(X,W.T))
    W_Tmp = lr*(Y-O.T).dot(X)
    W= W+W_Tmp
if __name__ == '__main__':
    for i in range(100):
        update()
        O = np.sign(np.dot(X,W.T))
        print(O)
        print(Y)
        if (O == Y).all():
            print("finished")
            print('epoch:',n)
            break
        x1 = [3, 4]
        y1 = [3, 3]
        x2 = [1]
        y2 = [1]
        k = -W[1] / W[2]
        d = -W[0] / W[2]
        print('k=', k)
        print('d=', d)
        xdata = np.linspace(0, 5)
        plt.figure()
        plt.plot(xdata, xdata * k + d, 'r')
        plt.plot(x1, y1, 'bo')
        plt.plot(x2, y2, 'yo')
        plt.ion()
        plt.show()