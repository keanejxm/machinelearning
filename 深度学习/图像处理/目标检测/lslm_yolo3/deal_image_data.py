#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  detection_yolo3.py
:time  2024/1/4 9:24
:desc  yolo3实现目标检测（基于dark-net的yolo3网络）
"""
import json
import os.path
import random

import numpy as np
import paddle.fluid as fluid
from PIL import Image, ImageEnhance, ImageDraw
from common_utils import *


# 算法；准备数据、构建模型、训练模型

class DealData:
    def __init__(self):
        self._filepath = DATA_PATH
        self._train_path = f"{self._filepath}/lslm"
        self._testpath = f"{self._filepath}/lslm-test"

    # ------------------------------------------准备数据----------------------------------------------------

    # 获取训练集标签参数
    def fetch_label_params(self):
        """
        获取训练集标签参数
        :return:
        """
        train_params = dict()
        train_path = f"{self._train_path}/label_list.txt"
        with open(train_path, "r", encoding="utf8") as r:
            for i in r.readlines():
                labels = i.strip().split(" ")
                train_params[labels[1]] = labels[0]
        return train_params

    # 坐标转换，由[x1,y1,w,h]转换为[center_x,center_y,w,h],并转为为范围在[0,1]之间的相对坐标
    @staticmethod
    def box_to_center_relative(box, img_height, img_width):
        """
        坐标转换
        :param box:
        :param img_height:
        :param img_width:
        :return:
        """
        assert len(box) == 4, "box should be a len(4) list or tuple"
        x, y, w, h = box
        x1 = max(x, 0)
        x2 = min(x + w - 1, img_width - 1)
        y1 = max(y, 0)
        y2 = min(y + h - 1, img_height - 1)

        x = (x1 + x2) / 2 / img_width
        y = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        return np.array([x, y, w, h])

    # ----------------图像增加：对比度，饱和读，明暗，颜色，扩张----------
    @staticmethod
    def random_brightness(img):  # 亮度
        prob = np.random.uniform(0, 1)
        # todo:图像扭曲
        if prob < 0.5:
            brightness_delta = 0.125
            delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        return img

    @staticmethod
    def random_contrast(img):  # 对比度
        prob = np.random.uniform(0, 1)
        # todo:
        if prob < 0.5:
            contrast_delta = 0.5
            delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        return img

    @staticmethod
    def random_saturation(img):  # 饱和度
        prob = np.random.uniform(0, 1)
        if prob < 0.5:
            saturation_delta = 0.5
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        return img

    @staticmethod
    def random_hue(img):  # 色调
        prob = np.random.uniform(0, 1)
        if prob < 0.5:
            hue_delta = 18
            delta = np.random.uniform(-hue_delta, hue_delta)
            img_hsv = np.array(img.convert("HSV"))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode="HSV").convert("RGB")
        return img

    def distort_image(self, img):
        # 图像扭曲
        prob = np.random.uniform(0, 1)
        if prob > 0.5:
            img = self.random_brightness(img)
            img = self.random_contrast(img)
            img = self.random_saturation(img)
            img = self.random_hue(img)
        else:
            img = self.random_brightness(img)
            img = self.random_contrast(img)
            img = self.random_saturation(img)
            img = self.random_hue(img)
        return img

    # 扩展增强
    def random_expand(self, img, gtboxes, keep_ratio=True):
        # todo 扩展参数
        if np.random.uniform(0, 1) < 0.5:
            return img, gtboxes
        max_ratio = 4
        w, h = img.size
        c = 3
        ratio_x = random.uniform(1, max_ratio)
        if keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, max_ratio)
        oh = int(h * ratio_y)
        ow = int(w * ratio_x)
        off_x = random.randint(0, ow - w)
        off_y = random.randint(0, oh - h)

        out_img = np.zeros((oh, ow, c), np.uint8)
        mean_rgb = [127.5, 127.5, 127.5]
        for i in range(c):
            # todo:数据增强使用的灰度值"mean_rgb": [127.5, 127.5, 127.5],  # 数据增强使用的灰度值
            out_img[:, :, i] = mean_rgb[i]
        out_img[off_y:off_y + h, off_x:off_x + w, :] = img
        gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
        gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
        gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
        gtboxes[:, 3] = gtboxes[:, 3] / ratio_y
        return Image.fromarray(out_img), gtboxes

    # 计算交并比
    def box_iou_xywh(self, box1, box2):
        assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4"
        assert box2.shape[-1] == 4, "Box1 shape[-1] should be 4"

        # 取两个框的坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        inter_w = inter_x2 - inter_x1 + 1  # 相交部分宽度
        inter_h = inter_y2 - inter_y1 + 1  # 相交部分高度
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0

        inter_area = inter_w * inter_h  # 相交面积
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)  # 框1的面积
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)  # 框2的面积

        return inter_area / (b1_area + b2_area - inter_area)  # 相集面积/并集面积

    # box裁剪
    def box_crop(self, boxes, labels, crop, img_shape):
        x, y, w, h = map(float, crop)
        im_w, im_h = map(float, img_shape)

        boxes = boxes.copy()
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

        crop_box = np.array([x, y, x + w, y + h])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

        boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
        boxes[:, :2] -= crop_box[:2]
        boxes[:, 2:] -= crop_box[:2]

        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
        boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
        labels = labels * mask.astype('float32')
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

        return boxes, labels, mask.sum()

    # 随机剪裁
    def random_crop(self, img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
        """"""
        if random.random() > 0.6:
            return img, boxes, labels
        if len(boxes) == 0:
            return img, boxes, labels
        if not constraints:
            # 最小/最大交并比值
            constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0),
            ]
        w, h = img.size
        crops = [(0, 0, w, h)]
        for min_iou, max_iou in constraints:
            for _ in range(max_trial):
                scale = random.uniform(scales[0], scales[1])
                aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), min(max_ratio, 1 / scale / scale))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                crop_box = np.array([[
                    (crop_x + crop_w / 2.0) / w,
                    (crop_y + crop_h / 2.0) / h,
                    crop_w / float(w),
                    crop_h / float(h)
                ]])
                iou = self.box_iou_xywh(crop_box, boxes)
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_w, crop_h))
                    break

        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, box_num = self.box_crop(boxes, labels, crop, (w, h))
            if box_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                            crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
            return img, crop_boxes, crop_labels
        return img, boxes, labels

    # 调整图像大小
    def resize_img(self, img, sampled_labels, input_size):
        target_size = input_size
        img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)  # 重置大小，双线性插值
        return img

    # 预处理
    def preprocess(self, img, bbox_labels, input_size):
        img_width, img_height = img.size
        sample_labels = np.array(bbox_labels)
        # TODO:是否扭曲增强
        img = self.distort_image(img)
        img, gtboxes = self.random_expand(img, sample_labels[:, 1:5])  # 扩展增强
        img, gtboxes, gtlabels = self.random_crop(img, gtboxes, sample_labels[:, 0])  # 随机剪裁
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
        img = self.resize_img(img, sample_labels, input_size)
        img = np.array(img).astype('float32')
        img -= [127.5, 127.5, 127.5]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img *= 0.007843
        return img, sample_labels

    # 读取画框后的文件数据
    def read_img(self):
        # label特征值
        label_params = self.fetch_label_params()
        train_path = f"{self._train_path}/train.txt"
        with open(train_path, "r", encoding="utf8") as r:
            file_list = [i for i in r.readlines()]
        for line in file_list:
            parts = line.split("\t")
            img = Image.open(os.path.join(self._train_path, parts[0]))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_width, img_height = img.size
            bbox_labels = []
            for bbox_str in parts[1:]:
                if len(bbox_str) <= 1:
                    continue
                bbox_sample = []
                bbox_json = json.loads(bbox_str)
                bbox_sample.append(float(label_params[bbox_json["value"]]))
                bbox = bbox_json["coordinate"]  # [[769.459, 241.819], [947.546, 506.167]] 0：框左上角坐标，1：框右下角坐标
                # 计算x,y,w,h
                box = [bbox[0][0], bbox[0][1], bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                # 转换
                bbox = self.box_to_center_relative(box, img_height=img_height, img_width=img_width)
                bbox_sample.append(float(bbox[0]))
                bbox_sample.append(float(bbox[1]))
                bbox_sample.append(float(bbox[2]))
                bbox_sample.append(float(bbox[3]))
                difficult = float(0)
                bbox_sample.append(difficult)
                bbox_labels.append(bbox_sample)
            if len(bbox_labels) == 0:
                continue
            input_size = [3, 448, 448]
            img, sample_labels = self.preprocess(img, bbox_labels, input_size)
            if len(sample_labels) == 0:
                continue

            boxes = sample_labels[:, 1:5]  # 坐标
            lbls = sample_labels[:, 0].astype('int32')  # 标签
            difficults = sample_labels[:, -1].astype('int32')
            max_box_num = 20  # 一副图像最多多少个目标物体
            cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)  # 控制最大目标数量
            ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
            ret_lbls = np.zeros((max_box_num), dtype=np.int32)
            ret_difficults = np.zeros((max_box_num), dtype=np.int32)
            ret_boxes[0: cope_size] = boxes[0: cope_size]
            ret_lbls[0: cope_size] = lbls[0: cope_size]
            ret_difficults[0: cope_size] = difficults[0: cope_size]

            yield img, ret_boxes, ret_lbls
