#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  write_figure_recognize.py
:time  2023/3/31 11:16
:desc  手写数字识别
"""
import os

import numpy as np
import operator
import matplotlib.pyplot as plt


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


def img_to_vector(file_path):
    """
    将图像转为向量
    :return:
    """
    vector_img = np.zeros((1, 1024))
    with open(file_path, "r") as r:
        try:
            img_res = r.readlines()
        finally:
            r.close()
        for i, row_value in enumerate(img_res):
            for j, column_value in enumerate(row_value.rstrip()):
                vector_img[0, 32 * i + j] = column_value
        return vector_img


def write_figure_recognize():
    """

    :return:
    """
    file_path = r"E:\keane_data\algorithm\knn\trainingDigits"
    file_names = os.listdir(file_path)
    file_num = len(file_names)
    feature_file = np.zeros((file_num,1024))
    y_label = list()
    for i,file_name in enumerate(file_names):
        file_class, file_num = file_name.split("_")
        data_file_path = os.path.join(file_path,file_name)
        y_label.append(int(file_class))
        feature_file[i,:] = img_to_vector(data_file_path)
    print(feature_file,y_label)



write_figure_recognize()