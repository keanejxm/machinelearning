#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  knn_study.py
:time  2023/3/23 10:02
:desc  
"""
from decimal import Decimal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
1、概述：
    knn输入：实例的特征向量
    knn输出：实例的类别
    分类：分类时根据k个最近邻的训练实例的类别，通过多数表决等方式进行预测。
2、k值的选择、距离度量、分类决策规则是k近邻的基本三要素
3、工作原理：
    ·假设有一个带有标签的样本数据集（训练数据集），其中包含每条数据与其所属分类的对应关系
    ·输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较
        计算新数据与样本数据集中每条数据的距离
        对求得的所有距离进行排序（从小到大，越小表示月接近）
        取前k（k一般小于等于20）个样本数据对应的分类标签
    ·求k个数据中出现次数最多的分类标签最为新的数据的分类

"""

# 约会数据路径
appoint_data_path = r"E:\data\algorithm\knn\datingTestSet2.txt"


def appoint_matrix():
    """
    将数据集转为Numpy类型，并分开特征向量与目标向量
    :return:
    """
    with open(appoint_data_path, "r") as r:
        try:
            appoint_data = r.readlines()
        finally:
            r.close()
        x_feature = list()
        y_label = list()
        for row_data in appoint_data:
            row_data = row_data.split("\t")
            x = [Decimal(i) for i in row_data[0:-1]]
            y = row_data[-1].rstrip()
            x_feature.append(x)
            y_label.append(int(y))
        x_feature = np.array(x_feature)
        y_label = np.array(y_label)
        return x_feature, y_label


def show_plt():
    """
    展示散点图
    :return:
    """
    x_feature, y_label = appoint_matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_feature[:, 0], x_feature[:, 1], 15.0 * y_label, 15.0 * y_label)
    plt.show()


show_plt()
