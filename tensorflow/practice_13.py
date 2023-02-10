#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_13.py
:time  2023/2/10 17:40
:desc  通过给tf.Module添加属性的方法进行封装
"""
import tensorflow as tf
import numpy as np

mymodule = tf.Module()
mymodule.x = tf.Variable(0.0)


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_print(a):
    mymodule.x.assign_add(a)
    tf.print(mymodule.x)
    return (mymodule.x)


mymodule.add_print = add_print
mymodule.add_print(tf.constant(1.0)).numpy()
print(mymodule.variables)

# 保存模型
tf.saved_model.save(mymodule, "./data/mymodule", signatures={"serving_default": mymodule.add_print})

# 加载模型
mymodule2 = tf.saved_model.load("./data/mymodule")
mymodule2.add_print(tf.constant(5.0))
