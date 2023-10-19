#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  build_model.py
:time  2023/2/27 9:23
:desc  构建模型的三种方法Squential按层顺序构建模型
"""

import pandas as pd
import numpy as np
import tensorflow as tf

train_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\train_token.csv"
test_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\test_token.csv"

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20


# 构建管道
def parse_line(line):
    t = tf.strings.split(line, "\t")
    label = tf.reshape(tf.case(tf.strings.to_number(t[0]), tf.int32), (-1,))
    features = tf.cast(tf.strings.to_number(tf.strings.split(t[1], " ")), tf.int32)
    return (features, label)


ds_train = tf.data.TextLineDataset(filenames=[train_token_path]).map(
    parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE
).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.TextLineDataset(filenames=[test_token_path]).map(
    parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE
).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# 一、Squential按层顺序创建模型
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
)
model.add(
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')
)
model.add(
    tf.keras.layers.MaxPool1D(2)
)
model.add(
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')
)
model.add(
    tf.keras.layers.MaxPool1D(2)
)
model.add(
    tf.keras.layers.Flatten()
)
model.add(tf.keras.layers.Dense(1, activation="sigmod"))

model.compile(
    optimizer="Nadam",
    loss="binary_crossentropy",
    metric=["accuracy", "AUC"]
)  # 配置训练方法是告知，优化器、损失函数、和准确率评测标准
model.summary()  # 输出模型各层的参数情况

import datetime

baselogger = tf.keras.callbacks.BaseLogger(stateful_metrics=["AUC"])
logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(
    ds_train, validation_data=ds_test, epochs=6, callbacks=[baselogger, tensorboard_callback]
)
