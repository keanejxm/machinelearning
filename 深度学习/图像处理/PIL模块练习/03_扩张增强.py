#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  03_扩张增强.py
:time  2024/1/15 14:00
:desc  
"""
import numpy as np
from PIL import Image, ImageEnhance
from common_utils import *

from PIL import Image

image_path = f"{DATA_PATH}/data_test/lslm/63.jpg"
img = Image.open(image_path)
w, h = img.size
max_ratio = 4
oh = int(h * max_ratio)
ow = int(w * max_ratio)
off_x = np.random.randint(0, ow - w)
off_y = np.random.randint(0, oh - h)
# "mean_rgb": [127.5, 127.5, 127.5],  # 数据增强使用的灰度值
out_img = np.zeros((oh, ow, 3), np.uint8)
for i in range(3):
    mean_rgb = [127.5, 127.5, 127.5]
    out_img[:, :, i] = mean_rgb[i]
out_img[off_y:off_y + h, off_x:off_x + w, :] = img
# [c_x,c_y,w,h]
box = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
box[:, 0] = box[:, 0] + off_x
box[:, 1] = box[:, 1] + off_y
box[:, 2] = box[:, 2] / max_ratio
box[:, 3] = box[:, 3] / max_ratio

img = Image.fromarray(out_img).convert("RGB")
img.show()
