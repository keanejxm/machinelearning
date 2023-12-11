#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  08_paddle_fruits_vgg_demo.py
:time  2023/12/11 10:00
:desc  vgg卷积神经网络
"""
import os
import paddle
import paddle.fluid as fluid
from multiprocessing import cpu_count

from common_utils import *


# 读取数据、处理数据

class FruitsVGG:
    def __init__(self):
        self.name_dict = {
            "apple": 0,
            "banana": 1,
            "grape": 2,
            "orange": 3,
            "pear": 4
        }
        self.image_file_path = f"./fruits/fruit.txt"
        self.buffer_size = 1024

    # 处理图片数据
    def deal_image(self):
        """
        处理图片，将图片路径放在一个文件中
        :return:
        """
        with open(self.image_file_path, "w+") as w:
            try:
                fruits_path = f"{DATA_PATH}/fruits/"
                for fruit_name in os.listdir(fruits_path):
                    fruit_path = os.path.join(fruits_path, fruit_name)
                    for image_name in os.listdir(fruit_path):
                        image_path = os.path.join(fruit_path, image_name)
                        # 将图片地址和图片类别放入到一个文件中
                        fruit_code = self.name_dict[fruit_name]
                        line = f"{image_path}\t{fruit_code}\n"
                        w.write(line)
            finally:
                w.close()

    # paddle图片读取器
    @staticmethod
    def reader_image(image_path):
        """"""
        # 读取图片
        img = paddle.dataset.image.load_image(image_path, is_color=True)
        # 简单处理图片
        img = paddle.dataset.image.simple_transform(im=img,
                                                    resize_size=100,
                                                    crop_size=100,
                                                    is_train=True,
                                                    is_color=True)
        # 归一化
        img = img.astype("float32") / 255
        return img

    # 搭建paddle图片文件读取器
    def paddle_img_reader(self):
        """"""

        def paddle_reader(img_path, buffered_size=1024):
            def reader():
                with open(self.image_file_path, "r") as r:
                    for img_line in [i.strip() for i in r.readlines()]:
                        img_path, img_label = img_line.split("\t")
                        yield img_path, img_label

            return paddle.reader.xmap_readers(mapper=self.reader_image,
                                              reader=reader,
                                              process_num=cpu_count(),
                                              buffer_size=buffered_size)

        # 随机读取
        shuffle_reader = paddle.reader.shuffle(paddle_reader,
                                               buf_size=1300)
        # 分批次读取
        batch_reader = paddle.batch(shuffle_reader,
                                    batch_size=32)
        return batch_reader

    # 搭建vgg网络模型
    def vgg_model(self, img):
        """"""


    # 程序入口
    def start(self):
        """
        程序入口
        :return:
        """
        self.deal_image()



if __name__ == '__main__':
    obj = FruitsVGG()
    obj.start()
