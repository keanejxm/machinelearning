#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_02.py
:time  2023/11/30 15:47
:desc  创建张量
"""
import numpy as np
import tensorflow as tf

# 创建一维张量
tensor1d = tf.constant([1,1,2,3,4,5])

# 创建二维
tensor2d = tf.constant(np.arange(1,10).reshape(3,3))

# 创建0
# 创建1
# zero_like