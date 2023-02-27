#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_7.py
:time  2023/2/1 11:37
:desc  TensorFlow求导
"""
import tensorflow as tf
import numpy as np

# f(x) = a*x**2 + b*x +c

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
with tf.GradientTape() as tape1:
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape1.gradient(y, x)
print(dy_dx)
# 对常量张量也可以求导，需要增加watch
with tf.GradientTape() as tape2:
    tape2.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c
dy_dx, dy_da, dy_db, dy_dc = tape2.gradient(y, [x, a, b, c])
print(dy_da)
print(dy_dc)

# 求二阶导数
with tf.GradientTape() as tape4:
    with tf.GradientTape() as tape3:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape3.gradient(y, x)
dy2_dx2 = tape4.gradient(dy_dx, x)
print(dy2_dx2)


# 在Autograph中使用
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    # 自变量转换成tf.float32
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    return ((dy_dx, y))


tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))

# 利用梯度磁带和优化器求最小值
x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
tf.print("y=", y, "x=", x)

x = tf.Variable(0.0, name="x", dtype=tf.float32)


def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    optimizer.minimize(f, [x])
tf.print("y=", f(), "x=", x)

# 在autograph中完成最小值求解
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    for _ in range(1000):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
    y = a * tf.pow(x, 2) + b * x + c
    return y


tf.print(minimizef())
tf.print(x)

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)


@tf.function
def train(epoch):
    for _ in range(epoch):
        optimizer.minimize(f, [x])
    return (f())


tf.print(train(1000))
tf.print(x)
