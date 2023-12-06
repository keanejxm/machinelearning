#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  04_paddle_reader_demo.py
:time  2023/12/6 15:37
:desc  paddle读取器
"""
import paddle


def reader_creator(file_path):
    def reader():
        with open(file_path, "r") as f:
            for line in f.readlines():
                yield line.strip()

    return reader


reader = reader_creator("test.txt")  # 顺序读取器
# 随机读取器
shuffle_reader = paddle.reader.shuffle(reader, 10)
# 批量读取器
batch_reader = paddle.batch(shuffle_reader,4)
# for line in reader():
# for line in shuffle_reader():
for data in batch_reader():
    print(data)
