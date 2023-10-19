#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_22.py
:time  2023/2/24 15:21
:desc  构建模型的三种方法
"""
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.python.keras import *

DATA_PATH = "E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data"

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20

"""
1、对于顺序结果的模型，优先使用Sequential方法构建
2、如果模型有多输入或者多输出，或者模型需要共享权重，或者模型具有残差连接等非顺序结构，推荐使用函数式API创建
3、如无特定必要，尽可能避免使用Model子类化的方式构建模型，这种方式提供了极大的灵活性，但也有更大的概率出错
"""

train_token_path = f"{DATA_PATH}/imdb/train_token.csv"
test_token_path = f"{DATA_PATH}/imdb/test_token.csv"


# 构建管道
def parse_line(line):
    t = tf.strings.split(line, "\t")
    label = tf.reshape(tf.cast(tf.strings.to_number(t[0]), tf.int32), (-1,))
    features = tf.cast(tf.strings.to_number(tf.strings.split(t[1], " ")), tf.int32)
    return (features, label)


ds_train = tf.data.TextLineDataset(filenames=[train_token_path]).map(
    parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE
).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE)
ds_test = tf.data.TextLineDataset(filenames=[test_token_path]).map(
    parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE
).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE
)

# Sequential方法构建模型
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
)
model.add(
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu")
)
model.add(
    tf.keras.layers.MaxPool1D(2)
)
model.add(
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")
)
model.add(
    tf.keras.layers.MaxPool1D(2)
)
model.add(
    tf.keras.layers.Flatten()
)
model.add(
    tf.keras.layers.Dense(1, activation="sigmoid")
)

model.compile(optimizer="Nadam",
              loss="binary_crossentropy",
              metrics=["accuracy", "AUC"]
              )
# model.summary()
model.summary()

baselogger = tf.keras.callbacks.BaseLogger(stateful_metrics=["AUC"])
logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(ds_train, validation_data=ds_test ,epochs=6, callbacks=[baselogger, tensorboard_callback])
