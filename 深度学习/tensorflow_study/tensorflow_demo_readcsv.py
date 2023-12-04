#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_read_csv.py
:time  2023/12/4 9:52
:desc  
"""
import os

import tensorflow as tf
from common_utils import *


def csv_read(filelist):
    # 构建文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 构建csv reader,读取队列内容（一行）
    reader = tf.TextLineReader()
    k, v = reader.read(file_queue)
    # 对内容进行解码
    ## record_defaults:指定每一个样本的每一列的类型，指定默认值
    record = [["None"], ["None"]]
    example, label = tf.decode_csv(v, record_defaults=record)
    # 批处理
    # batch_size:跟队列大小无关，只决定本批次取多少数据
    example_bat, label_bat = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)
    return example_bat, label_bat


if __name__ == '__main__':
    # 找到文件，构造一个列表
    dir_name = f"{DATA_PATH}/test_data/"
    file_names = os.listdir(dir_name)
    file_list = []
    for f in file_names:
        file_list.append(os.path.join(dir_name, f))
    example, label = csv_read(file_list)

    # 开启session运行
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 定义线程协调器
        # 开启读取文件线程
        # 调用tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中
        # 供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([example, label]))
        # 回收线程
        coord.request_stop()
        coord.join(threads)
