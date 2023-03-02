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


data = load_data(DATA_PATH)
np_data = np.array(data)
x = np_data[:,0:2]
y = np_data[:,-1]
plt.scatter(x,y)
plt.show()

