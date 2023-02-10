#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_14.py
:time  2023/2/10 17:49
:desc  tf.keras中的模型和层都是继承tf.Module实现的，也具有变量管理和子模块管理功能。
"""
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics

print(issubclass(tf.keras.Model,tf.Module))
print(issubclass(tf.keras.layers.Layer,tf.Module))
print(issubclass(tf.keras.Model,tf.keras.layers.Layer))

model = models.Sequential()
model.add(layers.Dense(4,input_shape = (10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))
model.summary()