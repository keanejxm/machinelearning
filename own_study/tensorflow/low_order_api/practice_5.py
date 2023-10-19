#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_5.py
:time  2023/1/31 17:27
:desc  构建图tensorflow基础练习
"""
import tensorflow as tf
import numpy as np

i = tf.constant(1)  # tf.int32类型常量
l = tf.constant(1, dtype=tf.int64)  # tf.int64类型常量
f = tf.constant(1.23)  # tf.float32类型常量
d = tf.constant(3.14, dtype=tf.double)  # tf.double类型常量
s = tf.constant("hello world")  # tf.string类型常量
b = tf.constant(True)  # tf.bool类型常量

# 标量
scalar = tf.constant(True)  # 标量
print(tf.rank(scalar))  # tf.rank的作用和numpy的ndim方法相同

# 一维常量
vector = tf.constant([1.0, 2.0, 3.0, 4.0])
print(tf.rank(vector))  # 查看常量维度
print(np.ndim(vector.numpy()))
# 二维常量(矩阵)
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(tf.rank(matrix).numpy())
print(np.ndim(matrix))
# 三维常量
tensor3 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
print(tensor3)
print(tf.rank(tensor3))
# 四维常量
tensor4 = tf.constant([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                       [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]])
print(tensor4)
print(tf.rank(tensor4))
# 使用tf.cast改变张量的数据类型
h = tf.constant([123, 456], dtype=tf.int32)
f = tf.cast(h, dtype=tf.float32)
print(h.dtype, f.dtype)
# 使用numpy方法将TensorFlow张量转为numpy中的张量
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y.numpy())
print(y.shape)
# 字符串转为numpy时的张量转码
u= tf.constant(u"你好 世界")
print(u.numpy())
print(u.numpy().decode("utf8"))


# 变量
c = tf.constant([1.0,2.0])
print(c)
print(id(c))
c = c+tf.constant([1.0,2.0])
print(c)
print(id(c))

v = tf.Variable([1.0,2.0],name="v")
print(v)
print(id(v))
v.assign_add([1.0,2.0])
print(v)
print(id(v))