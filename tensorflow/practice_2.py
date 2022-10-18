#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_2.py
:time  2022/10/18 14:02
:desc  
"""
import os
from collections import Counter


def load(filename):
    X = []
    Y = []
    with open(filename)as r:
        lines = r.readlines()
    for line in lines:
        datas = line.split('\t')
        X.append([float(d) for d in datas[:3]])
        Y.append(datas[-1].strip())
    return X, Y


# 归一化
def guiyihua(X):
    # 获取每一列
    l1, l2, l3 = [], [], []
    for x in X:
        l1.append(x[0])
        l2.append(x[1])
        l3.append(x[2])
    # 获取每列的最大值和最小值
    max1, min1 = max(l1), min(l1)
    max2, min2 = max(l2), min(l2)
    max3, min3 = max(l3), min(l3)
    # 重新赋值每一列
    new_X = []
    for i in range(len(l1)):
        new_X.append([(l1[i] - min1) / (max1 - min1), (l2[i] - min2) / (max2 - min2), (l3[i] - min3) / (max3 - min3)])
    return new_X


# 分类
def classify(X, x, Y, k):
    dis = []
    # 2. 计算新样本到所有样本之间的距离  计算方式使用欧式距离
    for xx in X:
        dis.append(((xx[0] - x[0]) ** 2 + (xx[1] - x[1]) ** 2 + (
                xx[2] - x[2]) ** 2) ** 0.5)
    # 3. 找到距离新样本最近的前k个样本 k值的选择凭经验 3-15个 最好是奇数
    new_dis = sorted(dis)[:k]
    y = []
    for d in new_dis:
        index = dis.index(d)
        y.append(Y[index])
    # 4. 统计出现次数最多的那个类别
    # 5. 将该类别作为新样本的类别输出
    return Counter(y).most_common(1)[0][0]


if __name__ == '__main__':
    filename = f"{os.path.dirname(__file__)}/dating.txt"
    X, Y = load(filename)
    X = guiyihua(X)
    print(X)
    x = [48111, 9.134528, 0.728045]  # 3
    train_X = X[:900]
    test_X = X[900:]
    train_Y = Y[:900]
    test_Y = Y[900:]
    for k in [3, 5, 7, 9, 11]:
        count = 0
        for i, x in enumerate(test_X):
            y = classify(X, x, Y, k)
            if y == test_Y[i]:
                count += 1
        print(count / 100)
