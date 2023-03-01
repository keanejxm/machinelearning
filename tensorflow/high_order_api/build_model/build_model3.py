#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  build_model3.py
:time  2023/2/27 10:36
:desc  Model子类化创建自定义模型
"""

import tensorflow as tf

train_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\train_token.csv"
test_token_path = r"E:\keane_python\人工智能资料\eat_tensorflow2_in_30_days-master\data\imdb\test_token.csv"

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20



inputs = tf.keras.layers.Input(shape=[MAX_LEN])
x  = tf.keras.layers.Embedding(MAX_WORDS,7)(inputs)

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


# 先定义一个残差模块，为自定义Layer
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size,
                                            activation="relu", padding="same")
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size,
                                            activation="relu", padding="same")
        self.conv3 = tf.keras.layers.Conv1D(filter=input_shape[-1], kernel_size=self.kernel_size,
                                            activation="relu", padding="same")
        self.maxpool = tf.keras.layers.MaxPool1D(2)
        super(ResBlock, self).build(input_shape)  # 相当于设置self.built = true

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.Add()([inputs, x])
        x = self.maxpool(x)
        return x

    # 如果要让自定义的Layer通过FunctionalAPI 组合成模型时可以序列化，需要自定义get_config方法
    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


# 测试ResBlock
resblock = ResBlock(kernel_size=3)
resblock.build(input_shape=(None, 200, 7))
resblock.compute_output_shape(input_shape=(None, 200, 7))


# 自定义模型
class ImdbModel(tf.keras.models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(MAX_WORDS, 7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense =  tf.keras.layers.Dense(1,activation='sigmod')
        super(ImdbModel, self).build(input_shape)
    def call(self,x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        return (x)

tf.keras.backend.clear_session()
model = ImdbModel()
model.build(input_shape =(None,200))
model.summary()
model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

import datetime
logdir = "./tflogs/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(ds_train,validation_data = ds_test,
                    epochs = 6,callbacks=[tensorboard_callback])