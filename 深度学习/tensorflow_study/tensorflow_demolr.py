#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demolr.py
:time  2023/12/4 9:07
:desc  
"""
import os.path

import tensorflow as tf

# 创建数据
x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
y_true = tf.matmul(x, [[2.0]]) + 5.0

# 建立线性回归模型
# y=wx+b
weight = tf.Variable(tf.random_normal([1, 1]), name="w", trainable=True)
bias = tf.Variable(0.0, name="b", trainable=True)

y_predict = tf.matmul(x, weight) + bias

# 损失函数
loss = tf.reduce_mean(tf.square(y_true - y_predict))
# 计算损失函数最小值
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 收集损失值
tf.summary.scalar("losses", loss)
# 将所有信息保存到磁盘
merged = tf.summary.merge_all()
# 初始化变量
init_op = tf.global_variables_initializer()

# 保存模型
saver = tf.train.Saver()  # 实例化saver

with tf.Session() as sess:
    sess.run(init_op)
    print("weight:", weight.eval(), "bias:", bias.eval())

    # 指定文件
    fw = tf.summary.FileWriter("../summary/", graph=sess.graph)
    # 训练之前检查是否有训练的模型，如果有需要加载已训练的模型
    saver_path = "../model/liner_model/"
    if os.path.exists(f"{saver_path}checkpoint"):
        saver.restore(sess, f"{saver_path}")
    print("weight:", weight.eval(), "bias:", bias.eval())

    for i in range(500):
        sess.run(train_op)
        summary = sess.run(merged)
        fw.add_summary(summary, i)
        print("i:", i, "weight:", weight.eval(), "bias:", bias.eval())

    # 保存模型
    saver.save(sess, f"{saver_path}")
