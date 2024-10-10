#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_001.py
:time  2024/10/8 10:52
:desc  
"""
import tensorflow as tf

# 1、定义常量
hello = tf.constant("hello world")
a = tf.constant(1.0)
b = tf.constant(5.0)
c = tf.add(a, b)
d = tf.constant(2.0, name="ddd")

# 2、生成张量
tensor_zero = tf.zeros(shape=(3, 2), dtype=tf.float32, name="0000")
tensor_ones = tf.ones(shape=(3, 3), dtype=tf.float32, name="11111")
tensor_nd = tf.random_normal(shape=(3, 3), mean=1.0, stddev=2.0, dtype=tf.float32)

# 3、占位符
tensor_ph1 = tf.placeholder(dtype=tf.float32, shape=(2, 3))
tensor_ph2 = tf.placeholder(dtype=tf.float32, shape=(None, 3))
# static_ph2 = tensor_ph2.get_shape()
dynamic_ph2 = tf.shape(tensor_ph2)

# 执行
with tf.Session() as sess:
    hello = sess.run(hello)
    print(a.eval())
    print(sess.run(b))
    print(sess.run(c))
    print(c.graph)
    # 张量属性
    print(a.shape)
    print(d.name)
    print(d.dtype)
    print(d.op)
    # 执行生成张量 0,1，正态分布
    print(tensor_zero.eval())
    print(tensor_ones.eval())
    print(tensor_nd.eval())
    # 占位符的执行
    print(sess.run(tensor_ph1, feed_dict={tensor_ph1: [[2, 2, 2],
                                                       [3, 3, 3]]}))

    print(sess.run(tensor_ph2, feed_dict={tensor_ph2: [[2, 2, 2]]}))
    # 动态与静态
    print(sess.run(dynamic_ph2))

