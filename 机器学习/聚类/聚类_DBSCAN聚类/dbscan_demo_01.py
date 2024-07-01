#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 9:37
# @Author  : keane
# @Site    : 
# @File    : dbscan_demo_01.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sc
import sklearn.metrics as sm


class ClusterDbscan:
    def __init__(self):
        self.data_path = r"D:\keane_data\data_test\multiple3.txt"

    # 读取数据
    def read_data(self):
        """"""
        data = pd.read_csv(self.data_path, names=["x1", "x2"])
        return data

    # 创建模型
    @staticmethod
    def dbscan_model(data):
        """
        构建DBSCAN聚类模型
        Returns:

        """
        model = sc.DBSCAN(
            eps=0.5,  # 圆的半径
            min_samples=5,  # 最小样本数
        )
        model.fit(data)
        label = model.labels_
        # print(label)
        return label

    def plt_draw(self, data, label):
        """"""
        plt.figure("DBSCAN_Cluster")
        plt.scatter(data["x1"], data["x2"], c=label)
        plt.show()

    def assess_model(self, data, label):
        """
        模型评估：聚类的模型评估指标为：轮廓系数
        """
        sil_score = sm.silhouette_samples(
            data,
            label,
            metric="euclidean"  # 欧式距离
        )
        print(sil_score)

    def start(self):
        data = self.read_data()
        label = self.dbscan_model(data)
        self.assess_model(data,label)
        self.plt_draw(data, label)


if __name__ == '__main__':
    dbscan_obj = ClusterDbscan()
    dbscan_obj.start()
