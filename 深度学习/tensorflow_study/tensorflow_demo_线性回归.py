#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_线性回归.py
:time  2024/10/10 10:59
:desc  使用tensorflow实现线性回归
"""
import tensorflow as tf
import numpy as np

# 编造数据 y = 2x + 4
x_i = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
y_true = tf.matmul(x_i, [[2.0]]) + 5.0
# 定义权重
init_w = tf.ones(shape=(100, 1))
w = tf.Variable(initial_value=init_w, dtype=tf.float32)
# 定义偏置
init_b = tf.zeros(shape=(1,))
b = tf.Variable(initial_value=init_b, dtype=tf.float32)

# x占位符
x = tf.placeholder(dtype=tf.float32, shape=(None, 1))

# 函数
# pred_y = tf.matmul()

with tf.Session() as sess:
    sess.run(x_i, y_true)
