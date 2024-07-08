#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 10:32
# @Author  : keane
# @Site    : 
# @File    : svm_demo_01.py
# @Software: PyCharm
import pandas as pd
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm


# 1、什么是支持向量机  2、什么是最优边界   3、线性可分和线性不可分  4、核函数：线性核函数、多项式核函数、径向基核函数
class SvmClassify:
    def __init__(self):
        self.file_path = f"E:\keane_data\data_test\multiple2.txt"
        # 划分训练集，测试集
        x, y = self.read_data()
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(x, y, test_size=0.1)
        self.model = None

    def read_data(self):
        data = pd.read_csv(self.file_path, header=None, names=["x1", "x2", "y"])
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return x, y

    def build_model(self):
        """"""
        self.model = svm.SVC(kernel="linear")  # 线性核函数
        self.model = svm.SVC(kernel="poly", degree=3)  # 多项式核函数
        self.model = svm.SVC(kernel="rbf")  # 径向基核函数
        return self.model

    def train_model(self):
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def access_model(self):
        """评估模型"""
        pred_y = self.model.predict(self.test_x)
        # 准确率
        accuracy_score = sm.accuracy_score(self.test_y, pred_y)
        # 查准率
        precision_score = sm.precision_score(self.test_y, pred_y)
        # 召回率
        recall_score = sm.recall_score(self.test_y, pred_y)
        # f1得分
        f1_score = sm.f1_score(self.test_y, pred_y)
        return dict(accuracyScore=accuracy_score, precisionScore=precision_score, recallScore=recall_score,
                    f1Score=f1_score)

    def start(self):
        data = self.read_data()
        model = self.build_model()
        model = self.train_model()
        access_res = self.access_model()
        print(access_res)


if __name__ == '__main__':
    svm_obj = SvmClassify()
    svm_obj.start()
