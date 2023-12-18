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
            yield i

    return reader


reader = reader_creator(data)

buffer_reader = paddle.reader.shuffle(reader, buf_size=2)
batch_reader = paddle.batch(reader=buffer_reader, batch_size=2)
# for i in batch_reader():
#     print(data)

x = fluid.layers.data(name="x", shape=[3, 2, 2], dtype="float32")
z = fluid.layers.fc(x, size=1, act="relu")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())
feeder = fluid.DataFeeder(feed_list=[x], place=place)
for data2 in batch_reader():
    res = exe.run(program=fluid.default_main_program(),
                  feed=feeder.feed(data2),
                  fetch_list=[z])
    print(res)



# data = [np.arange(1, 13).reshape((3, 2, 2)).astype("float32")]
#
# def reader_creator(data1):
#     def reader():
#         for i in data1:
#             yield i
#     return reader
#
# reader = reader_creator(data)
# buffer_reader = paddle.reader.shuffle(reader, buf_size=1)
# batch_reader = paddle.batch(reader=buffer_reader, batch_size=1)()
#
# x = [fluid.layers.data(name="x{}".format(i), shape=[2, 2], dtype="float32") for i in range(3)]
# y = fluid.layers.fc(input=x, size=1, act='relu')
#
# place = fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())
# feeder = fluid.DataFeeder(feed_list=x, place=place)
#
# for data_batch in batch_reader:
#     res = exe.run(fluid.default_main_program(),
#                   feed=feeder.feed(data_batch),
#                   fetch_list=[y])
#     print(res)
