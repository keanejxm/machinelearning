#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  09_paddle_data_demo.py
:time  2023/12/12 9:00
:desc  
"""
import paddle
import paddle.fluid as fluid
import numpy as np

data = [np.arange(1, 13).astype("float32"), np.arange(2, 14).astype("float32")]


def reader_creator(data1):
    def reader():
        for i in data1:
            yield i, 1

    return reader


reader = reader_creator(data)

buffer_reader = paddle.reader.shuffle(reader, buf_size=2)
batch_reader = paddle.batch(reader=buffer_reader, batch_size=2)
# for i in batch_reader():
#     print(data)

x = fluid.layers.data(name="x", shape=[3, 2, 2], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="int64")
z = fluid.layers.fc(x, size=1, act="relu")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())
feeder = fluid.DataFeeder(feed_list=[x, y], place=place)  # 经过测试feed_list中至少需要2个值
for data2 in batch_reader():
    res = exe.run(program=fluid.default_main_program(),
                  feed=feeder.feed(data2),
                  fetch_list=[z])
    print(res)
