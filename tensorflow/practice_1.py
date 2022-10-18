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

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.square(tf.matmul(X, w) + b - y)
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])

