#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  test.py
:time  2023/3/23 11:05
:desc  
"""
from numpy import *


def file2matrix(filename):
    """
    Desc:
        导入训练数据
    parameters:
        filename: 数据文件路径
    return:
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如: zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


# datingDataMat, datingLabels = file2matrix(r"E:\data\algorithm\knn\datingTestSet2.txt")
# import matplotlib
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()
from decimal import Decimal
import numpy as np
a = [Decimal("2.0433"),Decimal("3.3333")]
b = np.array(a,dtype=np.float64)
print(a)