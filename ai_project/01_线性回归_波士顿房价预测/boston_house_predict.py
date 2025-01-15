#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  boston_house_predict.py
:time  2025/1/15 8:47
:desc  波士顿房价预测
     CRIM     per capita crime rate by town
     ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
     INDUS    proportion of non-retail business acres per town
     CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
     NOX      nitric oxides concentration (parts per 10 million)
     RM       average number of rooms per dwelling
     AGE      proportion of owner-occupied units built prior to 1940
     DIS      weighted distances to five Boston employment centres
     RAD      index of accessibility to radial highways
     TAX      full-value property-tax rate per $10,000
     PTRATIO  pupil-teacher ratio by town
     B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
     LSTAT    % lower status of the population
     MEDV     Median value of owner-occupied homes in $1000's
"""
import os
import pickle
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import sklearn.model_selection as ms
import sklearn.metrics as sm


class BostonPredict:
    def __init__(self):
        pass

    @staticmethod
    def download_data() -> pd.DataFrame:
        """
        下载数据
        :return:
        """
        url_data = "http://lib.stat.cmu.edu/datasets/boston"
        df_data = pd.read_csv(url_data, sep="\s+", skiprows=22, header=None)
        df_new = np.hstack((df_data.iloc[::2], df_data.iloc[1::2, :3]))
        columns_data = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
                        "MEDV"]
        # 保存波士顿房价文件
        df_new = pd.DataFrame(data=df_new, columns=columns_data)
        writer = pd.ExcelWriter("boston_data.xlsx", engine="openpyxl")
        df_new.to_excel(excel_writer=writer, index=False)
        writer.save()
        writer.close()
        return df_new

    def fetch_data(self):
        """
        读取房价数据
        :return:
        """
        filepath = os.path.join(os.path.dirname(__file__), "boston_data.xlsx")
        if os.path.exists(filepath):
            df_data = pd.read_excel(filepath)
        else:
            # 下载数据
            df_data = self.download_data()
        x_data = df_data.iloc[:, :-1]
        y_data = df_data.iloc[:, -1]
        x_train, x_test, y_train, y_test = ms.train_test_split(x_data, y_data, test_size=0.1)
        return x_train, x_test, y_train, y_test

    def create_model(
            self,
            model_type: str,
    ):
        """
        创建线性回归模型
        :model_type:lr->一元线性模型,rd->岭回归,la->拉索回归,pl->多元线性回归
        :return:
        """
        if model_type == "lr":
            model = lm.LinearRegression()
        elif model_type == "rd":
            # 岭回归
            model = lm.Ridge()
        elif model_type == "la":
            # 拉索回归
            model = lm.Lasso()
        elif model_type == "pl":
            # 多元线性回归
            model = pl.make_pipeline(
                sp.PolynomialFeatures(2),
                lm.LinearRegression()
            )
        else:
            raise ValueError("未选择正确的线性")
        return model

    def train_model(
            self,
            model,
            x,
            y
    ):
        """
        训练模型
        :param model:
        :param x:
        :param y:
        :return:
        """
        model.fit(x, y)
        return model

    def predict_model(
            self,
            model,
            x
    ):
        """
        预测数据
        :param model:
        :param x:
        :return:
        """
        y_pred = model.predict(x)
        return y_pred

    def evaluate_model(
            self,
            test_y,
            pred_y
    ):
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
        return dict(R2Score=r2_score, meanAbsError=mean_abs_diff, midAbsError=mid_abs_diff,
                    meanSquError=mean_squ_diff)

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

    def execute_project(self):
        x_train, x_test, y_train, y_test = self.fetch_data()
        for model_type in ["lr","rd","la","pl"]:
            model = self.create_model(model_type=model_type)
            model = self.train_model(model, x_train, y_train)
            y_pred = self.predict_model(model, x_test)
            res = self.evaluate_model(test_y=y_test, pred_y=y_pred)
            print(res)


if __name__ == '__main__':
    obj = BostonPredict()
    obj.execute_project()
