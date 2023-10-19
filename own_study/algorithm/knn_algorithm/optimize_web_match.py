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

# 数据路径
appoint_data_path = r"E:\keane_data\algorithm\knn\datingTestSet2.txt"


# 采集数据
# 处理数据
# 分析数据
# 选择算法
# 训练算法
# 处理数据

class Knn:
    def __init__(self):
        pass

    def _knowledge_point(self):
        """
        本类用到的的知识点
        :return:
        """
        # 创建ndarray(0)0矩阵,4行3列
        zero_matrix = np.zeros((4, 3))
        # 对ndarray的每一行进行赋值
        zero_matrix[0, :] = [1, 2, 3]
        # np中tile使用,对ndarray数据进行赋值（复制（m,n）m行n列）
        a = np.array([1, 2, 3, 4])
        b = np.tile(a, (2, 1))
        # 使用matplotlib画图
        fig = plt.figure()
        fig.add_subplot(111)
        x = [1, 2, 3]
        y = [2, 3, 4]
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def _read_txt(txt_path):
        """
        读取txt数据
        :return:
        """
        # 读取txt数据
        with open(txt_path, "r") as r:
            try:
                res = r.readlines()
            finally:
                r.close()
        return res

    def _deal_data(self):
        """
        处理数据
        :return:
        """
        appoint_data_path = r"E:\keane_data\algorithm\knn\datingTestSet2.txt"
        txt_res = self._read_txt(appoint_data_path)
        zero_matrix = np.zeros((len(txt_res), 3))
        y_label = list()
        for row_index, txt_row in enumerate(txt_res):
            txt_info = txt_row.split("\t")
            zero_matrix[row_index, :] = txt_info[0:-1]
            y_label.append(int(txt_info[-1]))
        return zero_matrix, y_label

    def analyze_data(self, x, y):
        """
        分析数据，使用matplotlib查看数据分布
        :return:
        """
        fig = plt.figure()
        fig.add_subplot(111)
        plt.scatter(x[:, 0], x[:, 1], 15.0 * np.array(y), 15.0 * np.array(y))
        plt.show()

    def start(self):
        matrix_feature, y_label = self._deal_data()
        # self.analyze_data(matrix_feature,y_label)
        min_vals = matrix_feature.min(0)
        max_vals = matrix_feature.max(0)
        self._knowledge_point()
        print(min_vals, max_vals)


if __name__ == '__main__':
    obj = Knn()
    obj.start()
