#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  lslm_deal_img_data.py
:time  2024/1/15 9:25
:desc  螺丝螺母处理图片数据
"""
import json
import random
import numpy as np
from PIL import Image, ImageEnhance
from common_utils import *


# 计算box的交并比
class BoxIou:

    @staticmethod
    def iou_xywh(boxes1, boxes2):
        """
        计算boxes格式为[[center_x,center_y,w,h]]的交并比
        :param boxes1:
        :param boxes2:
        :return:
        """
        assert boxes1.shape[-1] == 4, "Box1 shape[-1] should be 4."
        assert boxes2.shape[-1] == 4, "Box2 shape[-1] should be 4."
        # 将xywh格式转为xyxy格式
        b1_x1, b1_x2 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y1, b1_y2 = boxes1[:, 1] - boxes1[:, 3] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
        b2_x1, b2_x2 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y1, b2_y2 = boxes2[:, 1] - boxes2[:, 3] / 2, boxes2[:, 1] + boxes2[:, 3] / 2

        # 求交集
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        # 交集的宽和长
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0
        # 交集面积
        inter_area = inter_w * inter_h

        # b1和b2的面积
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)
        return iou

    @staticmethod
    def iou_xyxy(boxes1, boxes2):
        """
        计算boxes格式为[[x1,y1,x2,y2]]的交并比
        :param boxes1:
        :param boxes2:
        :return:
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        # 求交集
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        # 交集的宽和长
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        inter_w[inter_w < 0] = 0
        inter_h[inter_h < 0] = 0
        # 交集面积
        inter_area = inter_w * inter_h

        # b1和b2的面积
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)
        return iou


class OperateImg:
    # ---------------- 图片扭曲增强 ----------------
    @staticmethod
    def random_distort(img, prob_condition, delta_default, type_dis=None):
        """
        图片扭曲增强
        :param img:
        :param prob_condition: 扭曲概率(小于此概率，图片进行对应处理。相反不处理)
        :param delta_default: 调节的参数（参数越大，图片调节越明显）
        :param type_dis: 需要扭曲的类型（bright:亮度；contrast:对比度；saturation:饱和度；hue:色调）
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < prob_condition:
            delta = np.random.uniform(-delta_default, delta_default) + 1
            if type_dis == "bright":  # 亮度调节
                img = ImageEnhance.Brightness(img).enhance(
                    delta  # 调节参数
                )
            elif type_dis == "contrast":  # 对比度调节
                img = ImageEnhance.Contrast(img).enhance(
                    delta
                )
            elif type_dis == "saturation":  # 饱和度调节
                img = ImageEnhance.Color(img).enhance(
                    delta
                )
            elif type_dis == "hue":
                img_hsv = np.array(img.convert("HSV"))
                img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
                img = Image.fromarray(img_hsv, mode="HSV").convert("RGB")
            else:
                return img
        return img

    # ------------ 图片扩展 ---------------
    @staticmethod
    def random_expand(img, boxes, prob_expand, max_ratio, keep_ration=True):
        """
        图片扩展
        :param img:图片
        :param boxes: 需要识别的框[[x1,y1,x2,y2],[x1,y1,x2,y2]]
        :param prob_expand: 概率
        :param max_ratio:扩展的最大概率
        :param keep_ration:x,y扩展的的概率是否一样
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob < prob_expand:
            return img, boxes
        h, w, c = np.array(img).shape
        # 扩展的倍率
        ratio_x = np.random.uniform(1, max_ratio)
        ratio_y = ratio_x if keep_ration else np.random.uniform(1, max_ratio)

        # 扩展后的宽和高
        oh, ow = int(ratio_y * h), int(ratio_x * w)

        # 扩展图片开始坐标
        off_x, off_y = random.randint(0, ow - w), random.randint(0, oh - h)

        # 扩展后没有图像的部分转为灰色
        out_img = np.zeros((oh, ow, c), np.uint8)
        for i in range(c):
            out_img[:, :, i] = 127.5
        out_img[off_y:off_y + h, off_x:off_x + w, :] = img

        # 处理box  box[[x1,y1,x2,y2],[x1,y1,x2,y2]]
        boxes[:, 0] = boxes[:, 0] - off_x
        boxes[:, 2] = boxes[:, 2] - off_x
        boxes[:, 1] = boxes[:, 1] - off_y
        boxes[:, 3] = boxes[:, 3] - off_y
        return Image.fromarray(out_img), boxes

    # -------------- 图片随机剪裁 --------------
    @staticmethod
    def random_crop(img, boxes, labels, prob_crop, scales=None, max_ratio=2.0, constraints=None,
                    max_trial=50):
        """
        :param img:
        :param boxes: 目标检测框出目标的框[[x1,y1,x2,y2],[x1,y1,x2,y2]]
        :param labels:
        :param prob_crop:
        :param scales:
        :param max_ratio:
        :param constraints:
        :param max_trial:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if scales is None:
            scales = [0.3, 1.0]
        if prob < prob_crop:
            return img, boxes, labels
        if len(boxes) == 0:
            return img, boxes, labels
        if not constraints:
            constraints = [(0.1, 1.0),
                           (0.3, 1.0),
                           (0.5, 1.0),
                           (0.7, 1.0),
                           (0.9, 1.0),
                           (0.0, 1.0)]
        w, h = img.size
        crops = [(0, 0, w, h)]
        for min_iou, max_iou in constraints:
            for _ in range(max_trial):
                scale = np.random.uniform(scales[0], scales[1])
                aspect_ratio = np.random.uniform(max(1 / max_iou, scale * scale), min(max_ratio, 1 / scale / scale))
                # 剪切后的高和宽
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                # 剪切后x,y开始坐标
                crop_x = random.randrange(w - crop_w)
                crop_y = random.randrange(h - crop_h)
                # 剪切后的图片
                crop_box = np.array([[crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]])
                iou = BoxIou.iou_xyxy(crop_box, boxes)
                # 通过iou(交并比)判断切割后的图片中是否有检测（螺丝螺母）数据
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    break
        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, boxes_num = DealImgData.crop_box_xyxy(boxes, labels, crop)
            if boxes_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[2], crop[3])).resize(img.size, Image.LANCZOS)
            return img, crop_boxes, crop_labels
        return img, boxes, labels


class DealImgData:
    def __init__(self):
        # 参数
        self.img_param = self.img_params()

        self.filepath = DATA_PATH
        self.train_path = f"{DATA_PATH}/data_test/lslm"
        self.test_path = f"{DATA_PATH}/data_test/lslm-test"
        self.img_pos_path = f"{self.train_path}/train.txt"
        self.label_path = f"{self.train_path}/label_list.txt"
        self.img_param["numDict"] = self.label_params()
        self.img_param["classDim"] = len(self.label_params())
        self.file_list()


    @staticmethod
    def img_params():
        return {
            "useGPU": False,
            "inputSize": [3, 448, 448],  # 网络输入图片大小
            "isDistort": True,  # 是否扭曲
            "maxBoxNum": 20,
            "distortParam": {
                "bright": {"prob": 0.5, "delta": 0.125},
                "saturation": {"prob": 0.5, "delta": 0.5},
                "contrast": {"prob": 0.5, "delta": 0.5},
                "hue": {"prob": 0.5, "delta": 0.5},
            },  # 图片扭曲参数
            "expandParam": {"prob": 0.5, "maxRatio": 4},
            "anchors": [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240],  # 锚点??
            "anchorMask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],  # Anchor Box序号
            "classDim": -1,  # 类别数量（初始化的时候设置）
            "imgCount": -1,  # 训练集图片数量
            "saveModelDir": "./model/persis_model",  # 增量模型保存目录
            "pretrainedModelDir": "./model/pretrained_model",  # 预训练模型保存目录
            "inferenceModelDir": "./model/inference_model",  # 预测模型路径
            # "numEpochs": 80,
            "numEpochs": 1,
            "validThresh": 0.01,
            "nmsTopK": 300,
            "nmsPosK": 300,
            "nmsThresh": 0.45,  # 非最大值抑制阈值
            "numDict": {},  # 数字-名称对应字典
            "sgdStrategy": {  # 梯度下降配置
                "learningRate": 0.002,
                "lrEpochs": [30, 50, 65],  # 学习率衰减分段（3个数字分为4段）
                "lrDecay": [1, 0.5, 0.25, 0.1]  # 每段采用的学习率，对应lr_epochs参数4段
            },
            # "trainBatchSize": 32,  # 对于完整yolov3，每一批的训练样本不能太多，内存会炸掉；如果使用tiny，可以适当大一些
            "trainBatchSize": 2,  # 对于完整yolov3，每一批的训练样本不能太多，内存会炸掉；如果使用tiny，可以适当大一些
        }

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

    def file_list(self):
        with open(self.img_pos_path, "r") as r:
            img_infos = [i.strip() for i in r.readlines()]
            self.img_param["imgCount"] = len(img_infos)

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

    @staticmethod
    def crop_box_xyxy(boxes, labels, crop):
        """
        切哥割盒子
        :param boxes:
        :param labels:
        :param crop: [c_x,c_y,w,h]# 切割的图片大小
        :return:
        """
        x1, y1, x2, y2 = map(float, crop)
        w = x2 - x1
        h = y2 - y1
        #
        boxes = boxes.copy()

        # 剪切的盒子
        crop_box = np.array([x1, y1, x2, y2])
        # (x1,y1 +x2,y2)/2 坐标平均值
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        # 判断每个box的中心点是否在剪切的图片中
        mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

        # 找到切割后的框的坐标
        boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
        # 盒子减去切割后图片的x,y坐标，就是盒子在切割的图片内的坐标
        boxes[:, :2] -= crop_box[:2]
        boxes[:, 2:] -= crop_box[:2]

        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
        boxes = boxes * np.expand_dims(mask.astype("float32"), axis=1)
        labels = np.array(labels, dtype=np.float32)
        labels = labels * mask.astype("float32")

        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h
        # 返回的box格式为归一化的（x,y,w,h）
        return boxes, labels, mask.sum()

    @staticmethod
    def crop_box_xywh(boxes, labels, crop, img_shape):
        """
        切哥割盒子
        :param boxes:[c_x,c_y,w,h]
        :param labels:
        :param crop: [c_x,c_y,w,h]# 切割的图片大小
        :param img_shape:
        :return:
        """
        x, y, w, h = map(float, crop)
        img_w, img_h = map(float, img_shape)
        #
        boxes = boxes.copy()
        # 将[[center_x,center_y,w,h]]--->[[x1,y1,x2,y2]]
        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:2] / 2) * img_w, (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:3] / 2) * img_h, (boxes[:, 1] + boxes[:, 3] / 2) * img_h

        # 剪切的盒子
        crop_box = np.array([x, y, x + w, y + h])
        # (x1,y1 +x2,y2)/2 坐标平均值
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # 判断平均值是否在剪切的box（x1,y1）到（x2,y2）之间
        mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

        # 找到切割后的框的坐标
        boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
        boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
        boxes[:, :2] -= crop_box[:2]
        boxes[:, 2:] -= crop_box[:2]

        mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
        boxes = boxes * np.expand_dims(mask.astype("float32"), axis=1)
        labels = labels * mask.astype("float32")

        boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
        boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h
        # 返回的box格式为归一化的（x,y,w,h）
        return boxes, labels, mask.sum()

    def deal_box(self, img_boxes, img_width, img_height):
        """
        处理lslm框数据将[左上角，右下角]->[中心x,中心y,宽,高]
        :param img_boxes:
        :param img_width:
        :param img_height:
        :return:
        """
        for img_box in img_boxes:
            img_box_json = json.loads(img_box)
            box_class = self.img_param["numDict"][img_box_json["value"]]
            box_coordinate = img_box_json["coordinate"]
            # 处理坐标数据[左上角，右下角]，[中心点x，中心点y，宽，高]
            w = box_coordinate[1][0] - box_coordinate[0][0]
            h = box_coordinate[1][1] - box_coordinate[0][1]
            x = box_coordinate[0][0]
            y = box_coordinate[0][1]
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            yield [center_x, center_y, w / img_width, h / img_height], box_class

    # 对图片进行预处理
    def img_process(self, img_boxes, img, mode):
        """"""
        boxes = list()
        box_labels = list()
        for img_box in img_boxes:
            img_box_json = json.loads(img_box)
            box_label = self.img_param["numDict"][img_box_json["value"]]
            box_coordinate = img_box_json["coordinate"]
            # 处理坐标数据[左上角，右下角]二维数据转为1维数据
            box_x1, box_y1, box_x2, box_y2 \
                = box_coordinate[0][0], box_coordinate[0][1], box_coordinate[1][0], box_coordinate[1][1]
            boxes.append([box_x1, box_y1, box_x2, box_y2])
            box_labels.append(box_label)
        boxes = np.array(boxes)
        # 图片扭曲增强
        if mode == "train":
            if self.img_param["isDistort"]:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    for type_dis in ["bright", "contrast", "saturation", "hue"]:
                        prob_condition = self.img_param["distortParam"][type_dis]["prob"]
                        delta_default = self.img_param["distortParam"][type_dis]["delta"]
                        img = OperateImg.random_distort(
                            img,
                            prob_condition=prob_condition,
                            delta_default=delta_default,
                            type_dis=type_dis
                        )
                else:
                    for type_dis in ["bright", "saturation", "hue", "contrast", "hue"]:
                        prob_condition = self.img_param["distortParam"][type_dis]["prob"]
                        delta_default = self.img_param["distortParam"][type_dis]["delta"]
                        img = OperateImg.random_distort(
                            img,
                            prob_condition=prob_condition,
                            delta_default=delta_default,
                            type_dis=type_dis
                        )
            # 扩展增强
            prob_expand = self.img_param["expandParam"]["prob"]
            max_ratio = self.img_param["expandParam"]["maxRatio"]
            img, boxes = OperateImg.random_expand(img, boxes, prob_expand=prob_expand, max_ratio=max_ratio)
            # 随机剪裁
            img, boxes, box_labels = OperateImg.random_crop(img, boxes, box_labels, prob_expand)
        # 重置image大小，双线性插值法
        img = img.resize((self.img_param["inputSize"][1], self.img_param["inputSize"][2]), Image.BILINEAR)
        # 转为ndarray
        img = np.array(img).astype("float32")
        # 图片归一化
        img -= [127.5, 127.5, 127.5]
        img = img.transpose((2, 0, 1))  # HWC-->高宽维度 TO CHW维度高宽
        img *= 0.007843

        return img, boxes, box_labels

    def read_img(self, mode):
        """
        1、读取画框后的图片信息
        2、读取图片信息
        3、处理图片的数据
        :return:
        """
        for img_info_list in self.read_img_pos():
            img_name = img_info_list[0]
            img_boxes = img_info_list[1:]
            if mode == "train" or mode == "eval":
                img = Image.open(os.path.join(self.train_path, img_name))
                img = img if img.mode == "RGB" else img.convert("RGB")
                img, boxes, labels = self.img_process(img_boxes, img, "train")
                if len(labels) == 0:
                    continue
                # different = np.zeros(labels.shape).astype("int32")
                max_box_num = self.img_param['maxBoxNum']  # 一副图像最多多少个目标物体
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)  # 控制最大目标数量
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_labels = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_labels[0: cope_size] = labels[0: cope_size]
                # ret_difficults[0: cope_size] = different[0: cope_size]
                yield img, ret_boxes, ret_labels
            else:
                img_path = os.path.join(self.train_path, img_name)
                yield Image.open(img_path)


if __name__ == '__main__':
    obj = DealImgData()
    obj.read_img('train')
