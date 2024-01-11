#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  18_图像闭运算.py
:time  2024/1/11 15:34
:desc  图像闭运算
"""
import numpy as np

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/9.png")
cv2.imshow("img",img)
kernel = np.ones(shape=(3,3),
                 dtype=np.uint8)

res = cv2.morphologyEx(img,
                       cv2.MORPH_CLOSE,
                       kernel,
                       iterations=10
                       )

cv2.imshow("res",res)

cv2.waitKey()