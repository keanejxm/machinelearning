#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  扭曲增强.py
:time  2024/1/15 10:40
:desc  
"""
import numpy as np
from PIL import Image, ImageEnhance
from common_utils import *

from PIL import Image




image_path = f"{DATA_PATH}/data_test/lslm/63.jpg"
img = Image.open(image_path)
# img.show()
#亮度调节对应的图像亮度由小变大，返回原图多少倍的亮度图。
# img = ImageEnhance.Brightness(img).enhance(1.5)
# 对比度调节对比度：白色画面(最亮时)下的亮度除以黑色画面(最暗时)下的亮度。值较小时。由雾的感觉，值较大时更清晰，能简单去雾。
# img = ImageEnhance.Contrast(img).enhance(1.5)
# 饱和度调节色彩饱和度：彩度除以明度，指色彩的鲜艳程度，也称色彩的纯度。值较小时，图像颜色降低，值较大时图像颜色更鲜艳。
# img = ImageEnhance.Color(img).enhance(1.5)
# 锐度调节
# img = ImageEnhance.Sharpness(img).enhance(3)
# img.show()
# 亮度调节
