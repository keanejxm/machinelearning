#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  lslm_deal_img_data.py
:time  2024/1/15 9:25
:desc  螺丝螺母处理图片数据
"""
import json
import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from common_utils import *


class DealImgData:
    def __init__(self):
        # 参数
        self.input_size = [3, 448, 448]

        self.filepath = DATA_PATH
        self.train_path = f"{DATA_PATH}/data_test/lslm"
        self.test_path = f"{DATA_PATH}/data_test/lslm-test"
        self.img_pos_path = f"{self.train_path}/train.txt"
        self.label_path = f"{self.train_path}/label_list.txt"
        self.label_param = self.label_params()

    def label_params(self):
        """
        获取训练集标签参数
        :return:
        """
        label_dict = dict()
        with open(self.label_path, "r") as r:
            label_list = [i.strip() for i in r.readlines()]
            for label_str in label_list:
                label_info = label_str.split(" ")
                label_dict[label_info[1]] = label_info[0]
        return label_dict

    def read_img_pos(self):
        """
        读取训练文件，获取图片名字，图片框位置、图片类别
        :return:
        """
        with open(self.img_pos_path, "r") as r:
            img_infos = [i.strip() for i in r.readlines()]
            for img_info in img_infos:
                img_info_list = [img_info_s for img_info_s in img_info.split("\t") if img_info_s.strip()]
                yield img_info_list

    def deal_box(self, img_boxes, img_width, img_height):
        """
        处理lslm框数据
        :param img_boxes:
        :param img_width:
        :param img_height:
        :return:
        """

        for img_box in img_boxes:
            img_box_json = json.loads(img_box)
            box_class = self.label_param[img_box_json["value"]]
            box_coordinate = img_box_json["coordinate"]
            # 处理坐标数据[左上角，右下角]，[中心点x，中心点y，宽，高]
            w = box_coordinate[1][0] - box_coordinate[0][0]
            h = box_coordinate[1][1] - box_coordinate[0][1]
            x = box_coordinate[0][0]
            y = box_coordinate[0][1]
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            yield [center_x, center_y, w / img_width, h / img_height], box_class

    def read_img(self):
        """
        1、读取画框后的图片信息
        2、读取图片信息
        3、处理图片的数据
        :return:
        """
        for img_info_list in self.read_img_pos():
            img_name = img_info_list[0]
            img = Image.open(os.path.join(self.train_path, img_name))
            img = img if img.mode == "RGB" else img.convert("RGB")
            img_width = img.width
            img_height = img.height
            # 处理图片中螺丝螺母的坐标信息
            for img_boxes, box_class in self.deal_box(img_info_list[1:], img_width, img_height):
                print(img_boxes)


if __name__ == '__main__':
    obj = DealImgData()
    obj.read_img()
