#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_20.py
:time  2023/2/23 16:17
:desc  损失函数和正则化项
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers, models, losses, regularizers, constraints

# 损失函数和正则化项
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(
    layers.Dense(64, input_dim=64,
                 kernel_regularizer=regularizers.l2(0.01),
                 activity_regularizer=regularizers.l1(0.01),
                 kernel_constraint=constraints.MaxNorm(max_value=2, axis=0))
)
model.add(
    layers.Dense(10,
                 kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation="sigmoid")
)
model.compile(optimizer="rmsprop",
              loss=losses.sparse_categorical_crossentropy,metrics=["AUC"]
              )
model.summary()