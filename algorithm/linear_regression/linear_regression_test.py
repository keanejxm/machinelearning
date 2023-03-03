#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  linear_regression_test.py
:time  2023/3/1 16:57
:desc  线性回归模型的实现
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = r"E:\keane_python\data\regression"
DATA_PATH = f"{FILE_PATH}/data.txt"


def load_data(file_path: str) -> list:
    data = list()
    with open(file_path, "r") as r:
        try:
            res = r.readlines()
            for line in res:
                params_nums = [float(num) for num in line.strip().split("\t")]
                data.append(params_nums)
        finally:
            r.close()
    return data


def liner_regression(x, y):
    """
    简单线性回归公式：
            y = kx   y:为预测值  y^为实际值
    实际值与误测值之间存在误差，找到合适的公式（即使得误差最小的公式），误差和(&)：最小
            &平方= （y^-kx）的平方
    误差和最小，(y^-kx)的平方的最低点，对（y^-kx）的平方进行求导，得到关于参数k的公式，导数为0的点即为最小值点
    0 = -2xy^+2kx**   k = xy/x**  除法为乘以该矩阵的逆矩阵
    :return:
    """
    # x的平方，即x的转置乘以x,
    x_square = x.T.dot(x)
    # 判断矩阵是否可逆，判断此矩阵的行列式值是否为0，如果不是0，可逆

    det_val = np.linalg.det(x_square)  # 该矩阵行列式的值
    if det_val ==0.0:
        raise ValueError(f"x矩阵不可逆")
    # 求逆矩阵（求x_square的逆矩阵）
    inverse = np.linalg.inv(x_square)

    w = inverse.dot(x.T.dot(y.T))
    print(w)


data = load_data(DATA_PATH)
np_data = np.array(data)
x = np_data[:, 0:2]
y = np_data[:, -1]
liner_regression(x,y)
