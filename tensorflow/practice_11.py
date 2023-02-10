#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_11.py
:time  2023/2/7 9:52
:desc  Autograph的使用规范
"""

import numpy as np
import tensorflow as tf


@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)


@tf.function
def tf_random():
    a = tf.random.normal((3, 3))
    tf.print(a)


# tf_random每次执行都会有重新生成随机数。
# np_random每次执行都是一样的结果。

x = tf.Variable(1.0, dtype=tf.float32)


@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return x


outer_var()

# 放在里面将会报错
# @tf.function
# def inter_var():
#     x = tf.Variable(1.0, dtype=tf.float32)
#     x.assign_add(1.0)
#     tf.print(x)
#     return x
#
# inter_var()


tensor_list1 = []


def append_tensor1(x):
    tensor_list1.append(x)
    return tensor_list1


append_tensor1(tf.constant(5.0))
append_tensor1(tf.constant(6.0))
print(tensor_list1)
tensor_list2 = []


@tf.function
def append_tensor2(x):
    tensor_list2.append(x)
    return tensor_list2


append_tensor2(tf.constant(5.0))
append_tensor2(tf.constant(6.0))
print(tensor_list1)
