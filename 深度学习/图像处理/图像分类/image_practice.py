#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  image_practice.py
:time  2024/3/18 15:31
:desc  图像练习
"""
import os
import paddle
import paddle.fluid as fluid
from common_utils import DATA_PATH

cv_image_path = os.path.join(DATA_PATH, "opencv_img")
fruits_path = os.path.join(DATA_PATH, "fruits")

image_path = os.path.join(cv_image_path, "img_data")
image_path_1 = os.path.join(image_path, "1.png")

img = paddle.dataset.image.load_image(image_path_1, is_color=True)

img = paddle.dataset.image.simple_transform(
    im=img,
    resize_size=100,
    crop_size=100,
    is_train=True,
    is_color=True
)
img = img.flatten().astype("float32") / 255

print(img)

img = fluid.layers.data(
    name="image",
    shape=[3, 100, 100],
    dtype="float32"
)
label = fluid.layers.data(
    name="label",
    shape=[1],
    dtype="int64"
)


def v_model(img):
    ret = fluid.layers.conv2d(
        input=img,
        num_filters=4,
        filter_size=3,
        stride=1,
        padding=0,
        groups=1,
    )
    return ret


ret = v_model(img)
print(ret)

import numpy as np

a = np.zeros([3, 4])
print(a)

