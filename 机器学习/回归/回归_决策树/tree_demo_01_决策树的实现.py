#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 7:10
# @Author  : keane
# @Site    : 
# @File    : tree_demo_01_决策树的实现.py
# @Software: PyCharm
import sklearn.tree as st
# import sklearn.datasets as sd
# import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.metrics as sm
import sklearn.ensemble as se
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 使用决策树回归方式来实现决策树的预测
class BostonTree:
    def __init__(self):
        self.data, self.target = self.fetch_data()
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(
            self.data,
            self.target,
            test_size=0.2,
            random_state=7
        )

    @staticmethod
    def fetch_data():
        """
        采集波士顿的数据
        Returns:

        """
        # data = sd.load_boston()
        # print(data)
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        return data, target

    def model_assess(
            self,
            model,
    ):
        """
        模型评估
        Args:
            model:模型参数

        Returns:

        """
        pred_y = model.predict(self.test_x)
        # r2_score
        r2_score = sm.r2_score(self.test_y, pred_y)
        # 平均绝对误差
        mean_abs_error = sm.mean_absolute_error(self.test_y, y_pred=pred_y)
        # 中位数绝对偏差
        mid_abs_error = sm.median_absolute_error(self.test_y, pred_y)
        # 均方误差
        mean_squ_error = sm.mean_squared_error(self.test_y, pred_y)
        return dict(r2Score=r2_score, meanAbsError=mean_abs_error, midAbsError=mid_abs_error,
                    meanSquError=mean_squ_error)

    def decision_tree_model(self):
        """
        构建决策树模型
        Returns:

        """
        model = st.DecisionTreeRegressor(max_depth=4)
        model.fit(self.train_x, self.train_y)
        assess_res = self.model_assess(model)
        self.draw_plt(model)
        print(assess_res)

    def draw_plt(
            self,
            model
    ):
        """
        画出决策树的图
        Args:
            model:

        Returns:

        """
        plt.figure("决策树图", figsize=(20, 9))
        plt.title("决策树图")
        # feature_importance = model.feature_importances_
        # st.plot_tree(model,max_depth=4,fontsize=14,feature_names=boston_data.features)
        st.plot_tree(model, max_depth=4, fontsize=14)
        plt.show()

    def adaboost_model(self):
        """
        集合算法adaboost
        Returns:

        """
        tree_model = st.DecisionTreeRegressor(max_depth=4)
        model = se.AdaBoostRegressor(tree_model, n_estimators=100, random_state=7)
        model.fit(self.train_x, self.train_y)
        assess_res = self.model_assess(model)
        print(assess_res)

    def gbdt_model(self):
        """
        GBDT算法
        Returns:

        """
        model = se.GradientBoostingRegressor(max_depth=4, random_state=7, n_estimators=100, min_samples_split=2)
        model.fit(self.train_x, self.train_y)
        assess_res = self.model_assess(model)
        print(assess_res)

    def random_forest(self):
        """"""
        model = se.RandomForestRegressor()
        model.fit(self.train_x, self.train_y)
        assess_res = self.model_assess(model)
        print(assess_res)


if __name__ == '__main__':
    boston_obj = BostonTree()
    # boston_obj.decision_tree_model()
    boston_obj.adaboost_model()
    boston_obj.gbdt_model()
    boston_obj.random_forest()
