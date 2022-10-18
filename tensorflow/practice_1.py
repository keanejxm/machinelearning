#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice.py
:time  2022/10/17 17:26
:desc  
"""
import os

import tensorflow as tf

# 定义一个随机数
random_float = tf.random.uniform(shape=())
print(random_float)

# 定义一个有两个元素的零向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

# 定义两个2*2的向量矩阵
a = tf.constant([[1., 2], [3, 4]])
b = tf.constant([[5., 6.], [7., 8.]])
print(a)
print(a.shape)
print(a.dtype)
print(a.numpy())
print(b)
# 矩阵求和  注意：dtype数据类型不同不可相加
c = tf.add(a, b)
print(c)

# 计算矩阵乘积
d = tf.matmul(a, b)
print(d)

# 自动求导
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
# 计算y关于x的导数
y_grad = tape.gradient(y,x)
print(y,y_grad)

import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)


print(X_raw-X_raw.min())
print(X_raw.max()-X_raw.min())

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

a, b = 0, 0

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)
