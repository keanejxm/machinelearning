#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_6.py
:time  2023/2/1 10:44
:desc  TensorFlow三种计算图
"""

import tensorflow as tf

# 静态计算图
g = tf.compat.v1.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(name="x", shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name="y", shape=[], dtype=tf.string)
    z = tf.strings.join([x, y], name="join", separator=" ")
with tf.compat.v1.Session(graph=g) as sess:
    result = sess.run(fetches=z, feed_dict={x: "hello", y: "world"})
    print(result)

# 动态计算图
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x, y], separator=" ")
print(z)
tf.print(z)


def strjoin(x, y):
    z = tf.strings.join([x, y], separator=" ")
    print(z)
    tf.print(z)


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)


@tf.function
def strjoin_1(x,y):
    z = tf.strings.join([x, y], separator=" ")
    print(z)
    tf.print(z)
    return z


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)

# TensorFlow日志
import datetime
import os

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("data", "autograph", stamp)
writer = tf.summary.create_file_writer(logdir)
# 开启AutoGraph追踪
tf.summary.trace_on(graph=True, profiler=True)
# 执行AutoGraph
result = strjoin_1("hell0", "world")
# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="autograph",
        step=0,
        profiler_outdir=logdir
    )
