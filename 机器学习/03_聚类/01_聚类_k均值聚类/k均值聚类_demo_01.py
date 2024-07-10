#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/30 22:30
# @Author  : keane
# @Site    : 
# @File    : 聚类_k均值聚类.py
# @Software: PyCharm
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sc
import sklearn.metrics as sm

class KMean:
    def __init__(self):
        if os.path.exists(r"D:\keane_data\data_test\multiple3.txt"):
            self.data_path = r"D:\keane_data\data_test\multiple3.txt"
        else:
            self.data_path = r"E:\keane_data\data_test\multiple3.txt"

    def read_data(self):
        """
        读取数据
        Returns:

        """
        data = pd.read_csv(self.data_path, header=None, names=["x1", "x2"])
        # print(data)
        return data

    @staticmethod
    def k_mean_model(data):
        model = sc.KMeans(n_clusters=4)
        model.fit(data)
        label = model.labels_  # 分类类别
        centers = model.cluster_centers_
        print(centers)
        # print(label)
        return label, centers

    @staticmethod
    def plt_draw(data, label, centers):
        plt.figure("k_mean聚类")
        plt.title("聚类分布")
        plt.scatter(data["x1"], data["x2"], c=label)
        centers_df = pd.DataFrame(centers, columns=["x1", "x2"])
        plt.scatter(centers_df["x1"], centers_df["x2"], marker="+", linewidths=20)
        plt.show()

    def assess_model(self,data,label):
        sil_score = sm.silhouette_samples(
            data,
            label,
            metric="euclidean"
        )
        print(sil_score)

    def start(self):
        data = self.read_data()
        # self.plt_draw(data)
        label, centers = k_mean_obj.k_mean_model(data)
        self.assess_model(data,label)
        self.plt_draw(data, label, centers)


if __name__ == '__main__':
    k_mean_obj = KMean()
    k_mean_obj.start()
