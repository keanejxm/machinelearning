#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/25 10:49
# @Author  : keane
# @Site    : 决策树实现分类
# @File    : tree_demo_01.py
# @Software: PyCharm
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.ensemble as se


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
        """
        随机森林
        :param data:
        :return:
        """
        train_x = data.iloc[:, :-1]
        train_y = data.iloc[:, -1]
        model = se.RandomForestClassifier(max_depth=4)
        model.fit(train_x, train_y)
        return model

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
        true_y = self.encoders[6].inverse_transform(test_data.iloc[:,-1].values)
        pred_y = self.encoders[6].inverse_transform(pred_y)
        print("真实：",true_y)
        print("预测：",pred_y)

    def start(self):
        data = self.load_data()
        encode_data = self.data_pretreatment(data)
        model = self.make_model(encode_data)
        self.test_mode(model)


if __name__ == '__main__':
    car_obj = CarClassify()
    car_obj.start()
