#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  build_model2.py
:time  2023/2/27 10:22
:desc  函数式API创建任意结构模型
"""
import tensorflow as tf

train_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\train_token.csv"
test_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\test_token.csv"

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20


# 构建管道
def parse_line(line):
    t = tf.strings.split(line, "\t")
    label = tf.reshape(tf.cast(tf.strings.to_number(t[0]), tf.int32), (-1,))
    features = tf.cast(tf.strings.to_number(tf.strings.split(t[1], " ")), tf.int32)
    return (features, label)


ds_train = tf.data.TextLineDataset(filenames=[train_token_path]) \
    .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = tf.data.TextLineDataset(filenames=[test_token_path]) \
    .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()
inputs = tf.keras.layers.Input(shape=[MAX_LEN])
x = tf.keras.layers.Embedding(MAX_WORDS, 7)(inputs)

branch1 = tf.keras.layers.SeparableConv1D(64, 3, activation='relu')(x)
branch1 = tf.keras.layers.MaxPool1D(3)(branch1)
branch1 = tf.keras.layers.SeparableConv1D(32, 3, activation='relu')(branch1)
branch1 = tf.keras.layers.GlobalMaxPool1D()(branch1)

branch2 = tf.keras.layers.SeparableConv1D(64, 5, activation="relu")(x)
branch2 = tf.keras.layers.MaxPool1D(5)(branch2)
branch2 = tf.keras.layers.SeparableConv1D(32, 5, activation="relu")(branch2)
branch2 = tf.keras.layers.GlobalMaxPool1D()(branch2)
branch3 = tf.keras.layers.SeparableConv1D(64, 7, activation="relu")(x)
branch3 = tf.keras.layers.MaxPool1D(7)(branch3)
branch3 = tf.keras.layers.SeparableConv1D(32, 7, activation="relu")(branch3)
branch3 = tf.keras.layers.GlobalMaxPool1D()(branch3)
concat = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(concat)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='Nadam',
              loss='binary_crossentropy',
              metrics=['accuracy', "AUC"])
model.summary()

import datetime

logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(ds_train, validation_data=ds_test, epochs=6, callbacks=[tensorboard_callback])
