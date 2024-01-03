#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/3 9:56
# @Author  : keane
# @Site    : 
# @File    : tensorflow_demo1.py
# @Software: PyCharm

import tensorflow as tf
import os

####### 图

a = tf.constant(5.0)  # 定义张量a
b = tf.constant(1.0)  # 定义张量b
c = tf.add(a, b)

graph = tf.get_default_graph()
print(graph)

graph2 = tf.Graph()
print(graph2)

# 使用graph2图计算
with graph2.as_default():
    d = tf.constant(11.0)

with tf.Session(graph=graph2) as sess:
    # res = sess.run(c)
    # print(res)
    # print(sess.graph)
    # print(a.graph)
    print(b.graph)
    print(sess.run(d))
    print(d.graph)

### 张量属性
a = tf.constant(5.0)

with tf.Session() as sess:
    print(sess.run(a))
    print(a.name)
    print(a.dtype)
    print(a.shape)
    print(a.op)
    print(a.graph)

### 生成张量

# 生成值全为0的张量
tensor_zeros = tf.zeros(shape=[2, 3], dtype=tf.float32)

# 生成值全为1的张量
tensor_ones = tf.ones(shape=[2, 3], dtype=tf.float32)

# 创建正态分布张量
tensor_nd = tf.random_normal(shape=[10], mean=1.7, stddev=0.2, dtype=tf.float32)

# 生成和输出张量形状一样的张量，值全为1
tensor_zeros_like = tf.zeros_like(tensor_ones)

with tf.Session() as sess:
    print(tensor_zeros.eval())
    print(tensor_ones.eval())
    print(tensor_nd.eval())
    print(tensor_zeros_like.eval())

### 张量类型转换
tensor_ones = tf.ones(shape=[2, 3], dtype=tf.int32)
tensor_float = tf.constant([1.1, 2.2, 3.3])

with tf.Session() as sess:
    print(tf.cast(tensor_ones, tf.float32).eval())

### 占位符

plhd = tf.placeholder(tf.float32, shape=[2, 3])  # 2行三列的tensor
plhd2 = tf.placeholder(tf.float32, shape=[None, 3])  # N行3列的tensor

with tf.Session() as sess:
    d = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    print(sess.run(plhd, feed_dict={plhd: d}))
    print("shape:", plhd.shape)
    print("name:", plhd.name)
    print("graph:", plhd.graph)
    print("op:", plhd.op)
    print(sess.run(plhd2, feed_dict={plhd2: d}))

### 张量形状改变

pld = tf.placeholder(tf.float32, shape=[None, 3])
print(pld)
pld.set_shape([4, 3])

new_pld = tf.reshape(pld, [3, 4])
print(new_pld)
with tf.Session() as sess:
    pass

### 矩阵运算
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.constant([[4, 3], [3, 2]], dtype=tf.float32)

# 矩阵想加
x_add_y = tf.add(x, y)
# 矩阵相乘
x_mul_y = tf.matmul(x, y)
# 对数
log_x = tf.log(x)

# 计算张量某一维度之和
x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)

#
data = tf.constant([1, 2, 3, 4, 5,6, 7, 8, 9, 10],dtype=tf.float32)
segment_ids = tf.constant([0,0,0,1,1,2,2,2,2,2],dtype=tf.int32)
x_seg_sum = tf.segment_sum(data,segment_ids)

with tf.Session() as sess:
    print(x_add_y.eval())
    print(x_mul_y.eval())
    print(log_x.eval())
    print(x_sum_1.eval())
    print(x_seg_sum.eval())


### 变量

# 创建普通张量
a = tf.constant([1,2,3,4,5])
var = tf.Variable(tf.random_normal([2,3],mean = 0.0,stddev=1.0),name="variable")

# 变量必须显式初始化，初始化操作，不运行
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([a,var]))