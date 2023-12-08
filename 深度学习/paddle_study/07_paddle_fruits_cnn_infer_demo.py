#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  05_paddle_fruits_cnn_demo.py
:time  2023/12/7 14:36
:desc  水果分类
"""
import os

import numpy as np
# 构建模型
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt
from common_utils import *
# 测试
from PIL import Image


# 读取测试图像
def load_img(path):
    img = paddle.dataset.image.load_and_transform(path,
                                                  128, 128,
                                                  False)
    img = img.astype("float32") / 255.0
    return img


# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = "../model/fruits/"
# 加载模型
infer_prog, feed_vars, fetch_targets = fluid.io.load_inference_model(
    model_save_dir, infer_exe
)
test_img = "1111.png"
infer_imgs = []
infer_imgs.append(load_img(test_img))
infer_imgs = numpy.array(infer_imgs)
# 执行预测
params = {feed_vars[0]: infer_imgs}
result = infer_exe.run(infer_prog, feed=params, fetch_list=fetch_targets)
print(result[0][0])
name_dict = {
    "apple": 0,
    "banana": 1,
    "grape": 2,
    "orange": 3,
    "pear": 4
}
r = np.argmax(result[0][0])
for k,v in name_dict.items():
    if v ==r:
        print(k)
# 评估
