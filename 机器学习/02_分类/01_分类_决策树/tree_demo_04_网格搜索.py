#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/25 15:38
# @Author  : keane
# @Site    : 
# @File    : tree_demo_04_网格搜索.py
# @Software: PyCharm
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import matplotlib.pyplot as plt


class CarClassify:
    def __init__(self):
        self.data_path = f"D:\keane_data\data_test\car.txt"
        self.encoders = dict()

    # 加载数据
    def load_data(self):
        data = pd.read_csv(self.data_path, header=None)
        return data

    # 数据预处理(标签编码)
    def data_pretreatment(self, data):
        """标签编码"""
        for i in data:
            encoder = sp.LabelEncoder()
            res = encoder.fit_transform(data[i])
            self.encoders[i] = encoder
            data[i] = res
        return data

    # 创建模型
    def make_model(self, data):
        """"""
        train_x = data.iloc[:, :-1]
        train_y = data.iloc[:, -1]
        sub_model = se.RandomForestClassifier()
        # 网格搜索
        param_grid = {
            "max_depth":np.arange(1,10,1),
            "n_estimators":np.arange(1,700,100),
        }
        model = ms.GridSearchCV(sub_model,param_grid=param_grid,cv=5)
        model.fit(train_x, train_y)
        return model

    # 验证曲线
    def validation_curve(self, model, train_x, train_y):
        param_range = np.arange(0.1, 1, 0.1)
        train_size, train_scores, test_scores = ms.learning_curve(model,
                                                                  train_x, train_y,
                                                                  # groups=param,
                                                                  train_sizes=param_range,  # 样本数量占比
                                                                  cv=5)  # 交叉验证次数
        self.validation_visual(param_range, test_scores)
        print(test_scores)

    def validation_visual(self, param_range, test_scores):
        """"""
        plt.figure("学习曲线")
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(param_range, test_scores_mean, "o-")
        plt.show()

    # 测试模型
    def test_mode(self, model):
        test_data = [
            ["high", "med", "5more", "4", "big", "low", "unacc"],
            ["high", "high", "4", "4", "med", "med", "acc"],
            ["low", "low", "2", "4", "small", "high", "good"],
            ["low", "low", "3", "4", "med", "high", "vgood"],
        ]
        test_data = pd.DataFrame(test_data)
        for key, encoder in self.encoders.items():
            test_data[key] = encoder.transform(test_data[key])
        pred_y = model.predict(test_data.iloc[:, :-1])
        true_y = self.encoders[6].inverse_transform(test_data.iloc[:, -1].values)
        pred_y = self.encoders[6].inverse_transform(pred_y)
        print("真实：", true_y)
        print("预测：", pred_y)

    def start(self):
        data = self.load_data()
        encode_data = self.data_pretreatment(data)
        model = self.make_model(encode_data)

        self.test_mode(model)


if __name__ == '__main__':
    car_obj = CarClassify()
    car_obj.start()
