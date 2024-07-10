#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 9:37
# @Author  : keane
# @Site    : 
# @File    : dbscan_demo_01.py
# @Software: PyCharm
import os
import pandas as pd
import sklearn.cluster as sc
import matplotlib.pyplot as plt


class DBSCANClassify:
    def __init__(self):
        if os.path.exists(r"D:\keane_data\data_test\multiple3.txt"):
            self.data_path = r"D:\keane_data\data_test\multiple3.txt"
        else:
            self.data_path = r"E:\keane_data\data_test\multiple3.txt"

    def read_data(self):
        data = pd.read_csv(self.data_path, header=None, names=["x1", "x2"])
        return data

    def build_model(self, data):
        model = sc.DBSCAN(eps=0.6,min_samples=5)
        model.fit(data)
        label_ = model.labels_
        return label_

    def draw_plt(self, data, labels):
        plt.figure("DBSCAN 聚类")
        plt.title("DBSCAN聚类")
        plt.scatter(data["x1"], data["x2"], c=labels)
        plt.show()

    def start(self):
        data = self.read_data()
        labels = self.build_model(data)
        self.draw_plt(data, labels)


if __name__ == '__main__':
    dbscan_obj = DBSCANClassify()
    dbscan_obj.start()
