#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo.py
:time  2023/11/28 10:06
:desc  tensorflow练习
"""
"""
定义执行相分离
"""
import tensorflow as tf

# 创建张量
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

# 执行
# sess = tf.Session()
# res = sess.run(temp)
# print(res)

with tf.Session() as sess:
    res = sess.run(temp)
    print(res)
# 关闭
# sess.close()
