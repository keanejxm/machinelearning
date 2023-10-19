#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_15.py
:time  2023/2/13 9:16
:desc  数据管道
"""
import os
import tensorflow as tf
import numpy as np
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# 构建数据管道
# 1、通过Numpy array 构建数据管道

# 加载iris数据
iris = datasets.load_iris()
x = iris["data"]
y = iris["target"]
ds1 = tf.data.Dataset.from_tensor_slices((iris["data"], iris["target"]))
for features, label in ds1.take(5):
    print(features, label)

# 2、通过Pandas DataFrame构建数据管道
dfiris = pd.DataFrame(iris["data"], columns=iris.feature_names)
ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"), iris["target"]))
for features, label in ds2.take(5):
    print(features, label)

# 3、通过Python generator构建数据管道
image_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    "./data/cifar2/test/",
    target_size=(32, 32),
    batch_size=20,
    class_mode="binary"
)
class_dict = image_generator.class_indices
print(class_dict)


def generator():
    for features, label in image_generator:
        yield (features, label)


ds3 = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
plt.figure(figsize=(6, 6))
for i, (img, label) in enumerate(ds3.unbatch().take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
# 4、通过csv文件构建数据管道
ds4 = tf.data.experimental.make_csv_dataset(
    file_pattern=["./data/titanic/train.csv", "./data/titanic/test.csv"],
    batch_size=3,
    label_name="Survived",
    na_value="",
    num_epochs=1,
    ignore_errors=True
)
for data, label in ds4.take(2):
    print(data, label)
# 5、通过文本文件构建数据管道
ds5 = tf.data.TextLineDataset(
    filenames=["./data/titanic/train.csv", "./data/titanic/test.csv"],
).skip(1)
for line in ds5.take(2):
    print(line)
# 6、通过文件路径构建数据管道
ds6 = tf.data.Dataset.list_files(
    "./data/cifar2/train/*/*.jpg"
)
for file in ds6.take(5):
    print(file)


def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
    img = tf.image.resize(img, size)
    return (img, label)


for i, (img, label) in enumerate(ds6.map(load_image).take(2)):
    plt.figure(i)
    plt.imshow((img / 255.0).numpy())
    plt.title("label = %d" % label)
    plt.xticks([])
    plt.yticks([])


# 7、通过records构建数据管道
def create_tfrecords(inpath, outpath):
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            # img = tf.image.decode_image(img)
            # img = tf.image.encode_jpeg(img) #统一成jpeg格式压缩
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


create_tfrecords("./data/cifar2/test/", "./data/cifar2_test.tfrecords/")


def parse_example(proto):
    description = {'img_raw': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"])  # 注意此处为jpeg格式
    img = tf.image.resize(img, (32, 32))
    label = example["label"]
    return (img, label)


ds7 = tf.data.TFRecordDataset("./data/cifar2_test.tfrecords").map(parse_example).shuffle(3000)
plt.figure(figsize=(6, 6))
for i, (img, label) in enumerate(ds7.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow((img / 255.0).numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
