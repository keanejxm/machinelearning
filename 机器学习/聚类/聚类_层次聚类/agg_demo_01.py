#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 9:56
# @Author  : keane
# @Site    : 
# @File    : agg_demo_01.py
# @Software: PyCharm
import pandas as pd
import sklearn.metrics as sm
import sklearn.cluster as sc
import matplotlib.pyplot as plt


class ClusterAgg:
    def __init__(self):
        self.data_path = r"D:\keane_data\data_test\multiple3.txt"

    def read_data(self):
        """
        读取数据
        Returns:

        """
        data = pd.read_csv(self.data_path, header=None, names=["x1", "x2"])
        # print(data)
        return data

    def agg_model(self, data):
        """"""
        model = sc.AgglomerativeClustering(n_clusters=4)
        model.fit(data)
        label = model.labels_
        return label

    @staticmethod
    def plt_draw(data, labels):
        plt.figure("AggCluster")
        plt.scatter(data["x1"], data["x2"], c=labels)
        plt.show()

    def start(self):
        data = self.read_data()
        label = self.agg_model(data)
        self.plt_draw(data, label)


if __name__ == '__main__':
    agg_obj = ClusterAgg()
    agg_obj.start()
