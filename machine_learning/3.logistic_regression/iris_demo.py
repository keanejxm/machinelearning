#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  iris.py
:time  2023/11/3 9:03
:desc  
"""
import pandas as pd
import sklearn.datasets as sd
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt

iris = sd.load_iris()

# 了解数据
# print(iris.keys())
# print(iris.data)
# print(iris.data.shape)
# print(iris.target)
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target
# print(data)

# 绘制萼片散点图
plt.scatter(data["sepal length (cm)"], data["sepal width (cm)"], c=data["target"], cmap="brg")
# 花瓣
plt.figure("花瓣")
plt.scatter(data["petal length (cm)"], data["petal width (cm)"], c=data["target"], cmap="brg")
# plt.show()

x = iris.data
y = iris.target

# 划分训练集测试集
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.3, random_state=7,stratify=y)

# 创建模型
model = lm.LogisticRegression(solver="liblinear")
model.fit(train_x, train_y)
# # 预测
# pred_y = model.predict(test_x)
#
# print(pred_y)
# # 评估
# # 准确率
# print((test_y == pred_y).sum() / test_y.size)
# print("精度", sm.accuracy_score(test_y, pred_y))
# # 查准率、召回率、f1得分  （每个类别都有自己的查准率、召回率、f1得分）
# print("查准率", sm.precision_score(test_y, pred_y, average="macro"))
# print("召回率", sm.recall_score(test_y, pred_y, average="macro"))
# print("f1分数", sm.f1_score(test_y, pred_y, average="macro"))
#
# # 混淆矩阵
# print("混淆矩阵","\n",sm.confusion_matrix(test_y,pred_y))
# # 分类报告
# print("分类报告",sm.classification_report(test_y,pred_y))

# 交叉验证
n = ms.cross_val_score(model,x,y,cv = 5,scoring="f1_weighted")
print(n.mean())