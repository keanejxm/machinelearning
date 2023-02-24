#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_17.py
:time  2023/2/13 13:54
:desc  提升管道性能
"""
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()
bb = iris["data"]
print(iris["data"])
dfiris = pd.DataFrame(iris["data"],columns = iris.feature_names)
aa = dfiris.to_dict("list")

ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"),iris["target"]))
for features,label in ds2.take(3):
    print(features,label)


# 使用prefetch方法让数据准备和参数迭代两个过程相互并行
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

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


# 数据准备和参数迭代两个过程默认情况下是串行的
# 模拟数据准备
def generator():
    for i in range(10):
        # 假设每次准备数据需要2s
        time.sleep(2)
        yield i


ds = tf.data.Dataset.from_generator(generator, output_types=(tf.int32))


# 模拟参数迭代
def train_step():
    # 假设每一步训练需要1s
    time.sleep(1)


printbar()
tf.print(tf.constant("start training...."))
for x in ds:
    train_step()
printbar()
tf.print(tf.constant("end training..."))

# 使用prefetch方法让数据准备和参数迭代两个过程相互并行
printbar()
tf.print(tf.constant("start training with prefetch"))
for x in ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE):
    train_step()
printbar()
tf.print(tf.constant("end training with prefetch"))

# 使用interleave方法可以让数据读取过程多进程并行，并将不同源数据夹在一起
ds_files = tf.data.Dataset.list_files("./data/titanic/*.csv")
ds = ds_files.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
for line in ds.take(4):
    print(line)
ds_files = tf.data.Dataset.list_files("./data/titanic/*.csv")
ds = ds_files.interleave(lambda x: tf.data.TextLineDataset(x).skip(1))
for line in ds.take(8):
    print(line)

# 使用map时设置num_parallel_calls让数据转换过程多进程进行
ds = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg")


def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)
    return (img, label)


# 单进程转换
printbar()
tf.print(tf.constant("start transformation..."))
ds_map = ds.map(load_image)
for _ in ds_map:
    pass
printbar()
tf.print(tf.constant("end transformation..."))

printbar()
tf.print(tf.constant("start parallel transformation..."))
ds_map_parallel = ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for _ in ds_map_parallel:
    pass
printbar()
tf.print(tf.constant("end parallel transformation..."))


# 使用cache方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大的情形
# 模拟数据准备
def generator():
    for i in range(5):
        # 假设每次准备数据需要2s
        time.sleep(2)
        yield i


ds = tf.data.Dataset.from_generator(generator, output_types=(tf.int32))
# 模拟训练
def train_step():
    pass
printbar()
tf.print(tf.constant("start training..."))
for epoch in tf.range(3):
    for x in ds:
        train_step()
    printbar()
    tf.print("epoch =",epoch," ended")
printbar()
tf.print(tf.constant("end training..."))

# 模拟数据准备
def generator():
    for i in range(5):
        #假设每次准备数据需要2s
        time.sleep(2)
        yield i
# 使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。
ds = tf.data.Dataset.from_generator(generator,output_types = (tf.int32)).cache()
# 模拟参数迭代
def train_step():
    #假设每一步训练需要0s
    time.sleep(0)
# 训练过程预计耗时 (5*2+5*0)+(5*0+5*0)*2 = 10s
printbar()
tf.print(tf.constant("start training..."))
for epoch in tf.range(3):
    for x in ds:
        train_step()
    printbar()
    tf.print("epoch =",epoch," ended")
printbar()
tf.print(tf.constant("end training..."))

# 使用map转换时，先batch，然后采用向量化的方法对每个batch进行转换
