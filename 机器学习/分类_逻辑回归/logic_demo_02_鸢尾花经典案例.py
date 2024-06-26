#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:54
# @Author  : keane
# @Site    : 
# @File    : logic_demo_02_鸢尾花经典案例.py
# @Software: PyCharm
import sklearn.datasets as sd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as sm


class LogicIris:
    def __init__(self):
        self.x, self.y = self.fetch_data()
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(self.x, self.y, test_size=0.1,
                                                                                   random_state=7)

    def fetch_data(self):
        """"""
        iris_data = sd.load_iris()
        x = iris_data.data
        y = iris_data.target
        return x, y

    def model_access(self, model):
        """
        模型评估
        Returns:

        """
        pred_y = model.predict(self.test_x)
        # 准确率
        accuracy_score = sm.accuracy_score(self.test_y, pred_y)
        # 差准率
        precision_score = sm.precision_score(self.test_y, pred_y, average="macro")
        # 召回率
        recall_score = sm.recall_score(self.test_y, pred_y, average="macro")
        # f1得分
        f1_score = sm.f1_score(self.test_y, pred_y, average="macro")
        return dict(accuracyScore=accuracy_score, precisionScore=precision_score, recallScore=recall_score,
                    f1Score=f1_score)

    def logic_model(self):
        model = lm.LogisticRegression(solver="liblinear")
        model.fit(self.train_x, self.train_y)
        pred_y = model.predict(self.test_x)
        # 准确率
        print((pred_y == self.test_y).sum() / self.test_y.size)
        print(self.model_access(model))


if __name__ == '__main__':
    iris_obj = LogicIris()
    iris_obj.logic_model()
