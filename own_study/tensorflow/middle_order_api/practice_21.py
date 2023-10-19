#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_21.py
:time  2023/2/24 11:14
:desc  优化器
"""

import tensorflow as tf
import numpy as np


# 打印时间分割线
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite), timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


# 求f(x) = a*x**2+b*x+c的最小值
# 使用optimizer.apply_gradients

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    while tf.constant(True):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
        # 迭代终止条件
        if tf.abs(dy_dx) < tf.constant(0.00001):
            break
        if tf.math.mod(optimizer.iterations, 100) == 0:
            printbar()
            tf.print("step = ", optimizer.iterations)
            tf.print("x=", x)
            tf.print("")
    y = a * tf.pow(x, 2) + b * x + c
    return y


tf.print("y = ", minimizef())
tf.print("x=", x)


# 使用optimizer.minimize
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return y


@tf.function
def train(epoch=1000):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    tf.print("epoch=", optimizer.iterations)
    return (f())


train(1000)
tf.print("y=", f())
tf.print("x = ", x)

# 使用model.fit
tf.keras.backend.clear_session()


class FakeModel(tf.keras.models.Model):
    def __init__(self, a, b, c):
        super(FakeModel, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def build(self):
        self.x = tf.Variable(0.0, name="x")
        self.built = True
    def call(self,features):
        loss = self.a*self.x**2+self.b*(self.x)+self.c
        return (tf.ones_like(features)*loss)

def myloss(y_true,y_pred):
    return tf.reduce_mean(y_pred)

model = FakeModel(tf.constant(1.0),tf.constant(-2.0),tf.constant(1.0))
model.build()
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),loss=myloss)
history = model.fit(tf.zeros((100,2)),
                    tf.ones(100),batch_size=1,epochs=10)
tf.print("x=",model.x)
tf.print("loss=",model(tf.constant(0.0)))