#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 10:33
# @Author  : keane
# @Site    : 
# @File    : nb_demo_01.py
# @Software: PyCharm
import pandas as pd
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import sklearn.metrics as sm


class NaiveBayes:
    def __init__(self):
        self.file_path = r"E:\keane_data\data_test\multiple1.txt"
        x, y = self.read_data()
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(x, y, test_size=0.1)

    def read_data(self):
        data = pd.read_csv(self.file_path, header=None, names=["x1", "x2", "y"])
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return x, y

    def build_model(self):
        model = nb.GaussianNB()  # 高斯贝叶斯
        # model = nb.BernoulliNB()  # 伯努利贝叶斯
        # model = nb.MultinomialNB()  # 多项式朴素贝叶斯
        model.fit(self.train_x, self.train_y)
        return model

    def access_model(self, model):
        y_pred = model.predict(self.test_x)
        # 准确率
        accuracy_score = sm.accuracy_score(self.test_y, y_pred)
        # 查准率
        precision_score = sm.precision_score(self.test_y, y_pred, average="micro")
        # 召回率
        recall_score = sm.recall_score(self.test_y, y_pred, average="micro")
        # f1得分
        f1_score = sm.f1_score(self.test_y, y_pred, average="micro")
        return dict(accuracyScore=accuracy_score, precisionScore=precision_score, recallScore=recall_score,
                    f1Score=f1_score)

    def start(self):
        model = self.build_model()
        access_res = self.access_model(model)
        print(access_res)

if __name__ == '__main__':
    nb_obj = NaiveBayes()
    nb_obj.start()
