#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_02.py
:time  2023/11/30 15:47
:desc  创建张量
"""
import numpy as np
import tensorflow as tf

# 创建一维张量
tensor1d = tf.constant([1,1,2,3,4,5])

# 创建二维
tensor2d = tf.constant(np.arange(1,10).reshape(3,3))

# 创建0
tensor_zeros = tf.zeros([2,3],dtype=tf.float32)
# 创建1
tensor_ones = tf.ones([2,3],dtype=tf.float32)
# zero_like
tensor_like = tf.zeros_like(tensor_ones)

# 矩阵相乘
x = tf.constant(np.arange(1,7).reshape([2,3]))
y = tf.constant(np.arange(1,13).reshape([3,4]))
z = tf.matmul(x,y)
a = tf.random_normal([100,1],mean=1,stddev=0.5,dtype=tf.float32)
b = tf.constant([[0.5]],dtype=tf.float32)
c = tf.matmul(a,b)
d = tf.Variable(1)
with tf.Session() as sess:
    print(sess.run(tensor_ones))
    print(tensor_like.eval())
    print(x.eval())
    print(y.eval())
    print(z.eval())
    print(a.eval())
    print(b.shape)
    print(b.eval())
    print(c.eval())
    print(d.shape)