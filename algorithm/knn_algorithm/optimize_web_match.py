#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  optimize_web_match.py
:time  2023/3/31 11:15
:desc  优化网站匹配knn案例1
"""

import numpy as np
import operator
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
appoint_data_path = r"E:\keane_data\algorithm\knn\datingTestSet2.txt"


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
        row_num = len(appoint_data)
        x_feature = np.zeros((row_num, 3))
        # x_feature = list()
        y_label = list()
        for i, row_data in enumerate(appoint_data):
            row_data = row_data.split("\t")
            x_feature[i, :] = row_data[0:-1]
            y = row_data[-1].rstrip()

            # x_feature.append(x)
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
    ax.scatter(x_feature[:, 0], x_feature[:, 1], s=15.0 * y_label, c=15.0 * y_label)
    plt.show()


"""
归一化：
    1、最小最大归一化
    2、Z—score归一化
"""


# 归一化特征值
def auto_normal(dataset):
    """
    归一化特征值
    :param dataset:
    :return:
    """
    # 计算每一个特征的最大值最小值
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    # 计算最大值、最小值的差
    diff_value = max_val - min_val
    normal_dataset = np.zeros(np.shape(dataset))
    m = normal_dataset.shape[0]
    # 生成与最小值之差组成的矩阵
    normal_dataset = dataset - np.tile(min_val, (m, 1))
    normal_dataset = normal_dataset / np.tile(diff_value, (m, 1))
    return normal_dataset, diff_value, min_val


def classify(inx, dataset, labels, k):
    """
    knn算法代码实现
    :param inx:
    :param dataset:
    :param label:
    :param k:
    :return:
    """
    dataset_size = dataset.shape[0]
    # 欧式距离公式 求差值，平法和，开方
    dif_val = np.tile(inx, (dataset_size, 1)) - dataset
    square_val = dif_val ** 2
    # 对每一行求和
    sum_square = square_val.sum(axis=1)
    distance = sum_square ** 0.5
    # 对距离进行排序，并返回索引（从小到大）
    sort_distance = distance.argsort()
    class_count = dict()
    for i in range(k):
        label_val = labels[sort_distance[i]]
        class_count[label_val] = class_count.get(label_val, 0) + 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


def dating_class_test():
    """"""

    x_feature, y_label = appoint_matrix()
    # 特质值归一化
    normal_dataset, diff_value, min_val = auto_normal(x_feature)
    # 得到数据行数
    m = normal_dataset.shape[0]
    # 设置测试样本数量
    test_ratio = 0.1
    test_num = int(m * test_ratio)
    error_count = 0
    for i in range(test_num):
        classify_result = classify(normal_dataset[i], normal_dataset[test_num:m, :], y_label[test_num:m], 3)
        if classify_result != y_label[i]: error_count += 1
    print(f"错误率{error_count / float(test_num)}")
    print(error_count)


def classify_person():
    """
    预测
    :return:
    """
    result_list = ['不喜欢', '一般', '喜欢']
    percent_tats = float(input("玩游戏所耗时间百分比"))
    ff_miles = float(input("飞行里程数"))
    ice_cream = float(input("消费冰淇淋"))
    dating_data_mat, dating_labels = appoint_matrix()
    norm_mat, ranges, min_vals = auto_normal(dating_data_mat)
    in_array = np.array([ff_miles, percent_tats, ice_cream])
    res = classify((in_array - min_vals) / ranges, norm_mat, dating_labels, 3)
    print(result_list[res - 1])


classify_person()
