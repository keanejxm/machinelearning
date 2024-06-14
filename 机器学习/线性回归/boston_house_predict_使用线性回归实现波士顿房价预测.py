#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  boston_house_predict.py
:time  2024/6/13 15:33
:desc  波士顿房价预测，线性回归
"""

import sklearn.datasets as sd
import sklearn.preprocessing as sp  # 预处理
import sklearn.linear_model as lm  # 线型模型
import sklearn.model_selection as ms  # 模型的选择和评估
import sklearn.metrics as sm  # 评估
import sklearn.pipeline as pl
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# data = sd.load_boston()
# print(data.feature_names)


class LinearRegression:
    def __init__(self):
        self.data, self.target = self.fetch_data()
        # 划分训练集测试集
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(self.data, self.target,
                                                                                   test_size=0.2)

    def fetch_data(self):
        """采集数据"""
        # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        data = self.rinse_data(data)
        return data, target

    @staticmethod
    def rinse_data(data):
        """清洗数据"""
        # 采集数据、清洗数据、构建模型、训练模型、模型评估、优化模型、应用
        data_mss = sp.MinMaxScaler()
        data_mss.fit(data)
        new_data = data_mss.transform(data)
        return new_data

    @staticmethod
    def model_assess(test_y, pred_y):
        """模型评估"""
        # 模型评估 回归模型的评估：
        # r2决定系数
        r2_score = sm.r2_score(test_y, pred_y)
        # 平均绝对误差
        mean_abs_diff = sm.mean_absolute_error(test_y, pred_y)
        # 中位数绝对偏差
        mid_abs_diff = sm.median_absolute_error(test_y, pred_y)
        # 均方误差
        mean_squ_diff = sm.mean_squared_error(test_y, pred_y)
        return dict(R2Score=r2_score, meanAbsError=mean_abs_diff, midAbsError=mid_abs_diff, meanSquError=mean_squ_diff)

    # 模型的保存
    @staticmethod
    def save_model(model_name, model):
        """保存模型"""
        with open(f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
            print("模型保存成功")

    @staticmethod
    def load_mode(model_path):
        """加载模型"""
        with open(model_path, "rb") as r:
            model = pickle.load(r)
            print("模型加载成功")
            return model

    # 画图
    # @staticmethod
    # def plt_draw(test_x, test_y, pred_y):
    #     """"""
    #     plt.figure("线性回归图", figsize=(12, 9), facecolor="lightgray")
    #     plt.title("线性回归")
    #     plt.xlabel("x", fontsize=12)
    #     plt.ylabel("y", fontsize=12)
    #     plt.scatter(test_x, test_y,label = "Sample")
    #     plt.plot(test_x, pred_y)
    #     plt.show()

    # 一元线性回归
    def linear_model(self):
        # 一元线性回归模型
        linear_model = lm.LinearRegression()
        linear_model.fit(self.train_x, self.train_y)
        pred_y = linear_model.predict(self.test_x)
        assess_value = self.model_assess(self.test_y, pred_y)
        return assess_value

    # 多元线性回归
    def multi_linear_model(self):
        """多元线型回归"""
        model = pl.make_pipeline(
            sp.PolynomialFeatures(2),  # polynomial:多项式
            lm.LinearRegression(),
        )
        model.fit(self.train_x, self.train_y)
        pred_y = model.predict(self.test_x)
        assess_value = self.model_assess(self.test_y, pred_y)
        # self.plt_draw(self.test_x,self.test_y,pred_y)
        return assess_value

    # 拉索回归
    def lasso_regression(self):
        model = lm.Lasso()
        model.fit(self.train_x, self.train_y)
        pred_y = model.predict(self.test_x)
        assess_value = self.model_assess(self.test_y, pred_y)
        return assess_value

    # 岭回归
    def ridge_regression(self):
        model = lm.Ridge()
        model.fit(self.train_x, self.train_y)
        pred_y = model.predict(self.test_x)
        assess_value = self.model_assess(self.test_y, pred_y)
        return assess_value


if __name__ == '__main__':
    lr_obj = LinearRegression()
    print(lr_obj.linear_model())
    print(lr_obj.multi_linear_model())
