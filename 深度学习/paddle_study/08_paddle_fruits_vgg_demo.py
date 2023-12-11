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
        img = img.flatten().astype("float32") / 255
        return img

    # 搭建paddle图片文件读取器
    def paddle_img_reader(self):
        """"""

        def paddle_reader(buffered_size=1024):
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
    def vgg_model(self, img, type_size):
        """"""

        # 卷积池化组
        def conv_pool(ipt, num_filter, groups, dropouts):
            return fluid.nets.img_conv_group(input=ipt,
                                             conv_filter_size=3,  # 卷积核大小
                                             conv_num_filter=[num_filter] * groups,  # 卷积核数量
                                             pool_size=2,
                                             pool_stride=2,
                                             conv_act="relu",
                                             conv_with_batchnorm=True,
                                             conv_batchnorm_drop_rate=dropouts,
                                             pool_type="max")  # 池化类型

        # 五个卷积池化组
        conv1 = conv_pool(img, num_filter=64, groups=2, dropouts=[0.0, 0])
        conv2 = conv_pool(conv1, num_filter=128, groups=2, dropouts=[0.0, 0])
        conv3 = conv_pool(conv2, num_filter=256, groups=3, dropouts=[0.0, 0.0, 0])
        conv4 = conv_pool(conv3, num_filter=512, groups=3, dropouts=[0.0, 0.0, 0])
        conv5 = conv_pool(conv4, num_filter=512, groups=3, dropouts=[0.0, 0.0, 0])
        # drop
        drop = fluid.layers.dropout(x=conv5, dropout_prob=0.2)
        # fc
        fc1 = fluid.layers.fc(drop, size=512, act=None)
        bn = fluid.layers.batch_norm(input=fc1, act="relu")
        drop1 = fluid.layers.dropout(x=bn, dropout_prob=0.2)
        fc2 = fluid.layers.fc(input=drop1, size=512, act=None)
        predict = fluid.layers.fc(input=fc2, size=type_size, act="softmax")
        return predict

    # 训练模型
    def train_model(self):
        """"""
        image = fluid.layers.data(name="x", shape=[3, 100, 100], dtype="float32")
        label = fluid.layers.data(name="y", shape=[1], dtype="int64")
        predict = self.vgg_model(img=image, type_size=5)
        cost = fluid.layers.cross_entropy(input=label,
                                          label=predict)
        avg_cost = fluid.layers.mean(cost)
        # 准确率
        accuracy = fluid.layers.accuracy(input=predict,
                                         label=label)
        # 优化器
        optimizer = fluid.optimizer.Adam(learning_rate=0.00001)
        optimizer.minimize()
        # 执行
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(program=fluid.default_startup_program())
        feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

        # 加载增量模型
        persis_model_dir = "persis_model"
        if os.path.exists(persis_model_dir):
            fluid.io.load_persistables(exe, persis_model_dir, fluid.default_main_program())

        for epoch in range(100):
            for batch_id, data in enumerate(self.paddle_img_reader()):
                train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                                feed=feeder.feed(data),
                                                fetch_list=[avg_cost, accuracy])
                if batch_id%20 ==0:
                    print(f"epoch:{epoch},batchId:{batch_id},cost:{train_cost},acc:{train_acc}")

    def fetch_img_list(self):
        """"""
        with open(self.image_file_path,"r") as r:
            for j in [i.strip() for i in r.readlines()]:
                img_path,label = j.split("\t")
                yield img_path,label
    # 程序入口
    def start(self):
        """
        程序入口
        :return:
        """
        self.deal_image()
        for img_path,label in self.fetch_img_list():
            self.reader_image(img_path)


if __name__ == '__main__':
    obj = FruitsVGG()
    obj.start()
