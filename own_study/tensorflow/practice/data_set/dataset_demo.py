#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  dataset_demo.py
:time  2023/3/6 11:16
:desc  
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# 正态分布
x = tf.random.normal(shape=(100, 1), mean=1.75, stddev=0.5)

# 输出y = 2x+5
y = tf.matmul(x, [[2.0]]) + 5.0

# 搭建模型，设置偏执
# 权重初始值
init_w = tf.random.normal([1, 1])
# 权重
weight = tf.Variable(init_w, trainable=True)
# 偏执
bias = tf.Variable(0.0, trainable=True)

# 预测函数
y_predict = tf.matmul(x, weight) + bias

# 损失值
loss= tf.reduce_mean(tf.square(y-y_predict))

#

# 画图

plt.plot(x, y)
plt.show()
