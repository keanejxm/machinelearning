#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_16.py
:time  2023/2/13 11:08
:desc  应用数据转换
"""
import os
import tensorflow as tf
import numpy as np
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# 应用数据转换
# map
ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
ds_map = ds.map(lambda x: tf.strings.split(x, " "))
for x in ds_map:
    print(x)
# flat_map:将转换函数映射到数据集的每一个元素，并将Dataset压平
ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
ds_flat_map = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for x in ds_flat_map:
    print(x)

# interleave:效果类似flat_map,但可以将不同来源是数据夹在一起
ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
da_interleave = ds.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for x in da_interleave:
    print(x)

# filter:过滤掉某些元素
ds = tf.data.Dataset.from_tensor_slices(["hello world", "hello china", "hello beijing"])
# 找出含有字母a或者B的元素
ds_filter = ds.filter(lambda x: tf.strings.regex_full_match(x, ".*[a|B].*"))
for x in ds_filter:
    print(x)

# zip:将两个长度相同的dataset横向铰合
ds1 = tf.data.Dataset.range(0, 3)
ds2 = tf.data.Dataset.range(3, 6)
ds3 = tf.data.Dataset.range(6, 9)
ds_zip = tf.data.Dataset.zip((ds1, ds2, ds3))
for x, y, z in ds_zip:
    print(x.numpy(), y.numpy(), z.numpy())

# concatenate:将两个Dataset纵向连接
ds1 = tf.data.Dataset.range(0, 3)
ds2 = tf.data.Dataset.range(3, 6)
ds_concat = tf.data.Dataset.concatenate(ds1, ds2)
for x in ds_concat:
    print(x)

# reduce:执行归并操作
ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5.0])
result = ds.reduce(0.0, lambda x, y: tf.add(x, y))

# batch:构建批次，每次放一个批次。比原始数据增加一个维度。其逆操作是unbatch
ds = tf.data.Dataset.range(12)
ds_batch = ds.batch(4)
for x in ds_batch:
    print(x)

# padded_batch:构建批次，类似batch,但可以填充到相同的形状
elements = [[1, 2], [3, 4, 5], [6, 7], [8]]
ds = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
ds_padded_batch = ds.padded_batch(2, padded_shapes=[4, ])
for x in ds_padded_batch:
    print(x)

# window:构建滑动窗口，返回Dataset of Dataset
ds = tf.data.Dataset.range(12)
ds_window = ds.window(3, shift=1).flat_map(lambda x: x.batch(3, drop_remainder=True))
for x in ds_window:
    print(x)

# shuffle:数据顺序洗牌
ds = tf.data.Dataset.range(12)
ds_shuffle = ds.shuffle(buffer_size=5)
for x in ds_shuffle:
    print(x)

# shard:采样，从某个位置开始隔固定距离采样一个元素
ds = tf.data.Dataset.range(12)
ds_shard = ds.shard(3,index=1)
for x in ds_shard:
    print(x)

# repeat:重复数据若干次，不带参数时，重复无数次

ds= tf.data.Dataset.range(3)
ds_repeat = ds.repeat(3)
for x in ds_repeat:
    print(x)

