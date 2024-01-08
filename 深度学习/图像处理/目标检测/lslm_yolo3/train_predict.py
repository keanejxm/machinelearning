#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  train_predict.py
:time  2024/1/5 15:12
:desc  模型的训练与预测
"""
import paddle.reader

from deal_image_data import DealData
import paddle.fluid as fluid


class TrainPredict:
    def __init__(self):
        pass

    # 读取器
    def paddle_data_reader(self):
        def reader():
            for img, ret_boxes, ret_lbls in DealData().read_img():
                yield img, ret_boxes, ret_lbls
        return reader

    def train_model(self):
        place = fluid.CPUPlace()
        # 创建2个program
        train_program = fluid.Program()
        start_program = fluid.Program()

        # "max_box_num": 20,  # 一幅图上最多有多少个目标
        max_box_num = 20
        with fluid.program_guard(train_program, start_program):
            img = fluid.layers.data(name="img", shape=[3, 448, 448], dtype="float32")
            gt_box = fluid.layers.data(name="gt_box", shape=[max_box_num, 4], dtype="float32")
            gt_label = fluid.layers.data(name="gt_label", shape=[max_box_num], dtype="int32")
            feeder = fluid.DataFeeder(feed_list=[img,gt_box,gt_label],place =place,program=train_program)
            reader = self.paddle_data_reader()
            reader = paddle.reader.shuffle(reader,buf_size=32)
            reader = paddle.batch(reader,batch_size=32)


    def predict_model(self):
        pass
