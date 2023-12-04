#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_read_img.py
:time  2023/12/4 11:03
:desc  
"""
import os

import tensorflow as tf
from common_utils import *


def img_read(filelist):
    """"""
    # 定义通道
    file_queue = tf.train.string_input_producer(filelist)
    # 定义reader
    reader = tf.WholeFileReader()
    k,v = reader.read(file_queue)

    # 解码
    # 定义解码格式
    img = tf.image.decode_jpeg(v)
    # 批处理，图片需要处理成统一大小
    img_resize = tf.image.resize(img,[200,200])
    img_resize.set_shape([200,200,3])# 固定样本形状，批处理时对数据形状有要求
    img_bat = tf.train.batch([img_resize],
                             batch_size=10,
                             num_threads=1)
    return img_bat

if __name__ == '__main__':
    dir_name = f"{DATA_PATH}/test_img/"
    file_names = os.listdir(dir_name)
    file_list = []
    for f in file_names:
        file_list.append(os.path.join(dir_name,f))
    img = img_read(file_list)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord=coord)
        print(sess.run(img))
        coord.request_stop()
        coord.join(threads)