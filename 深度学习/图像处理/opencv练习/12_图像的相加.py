#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  图像的相加.py
:time  2024/1/11 14:57
:desc  图像的相加：对应位置对应相加
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

lena = cv2.imread(f"{img_path}/lena.jpg", 0)
lily = cv2.imread(f"{img_path}/lily_square.png", 0)

cv2.imshow("lena", lena)
cv2.imshow("lily", lily)

# 相加
add = cv2.add(lena, lily)
cv2.imshow("add", add)

# 按照权重进行相加
add_w = cv2.addWeighted(
    lena, 0.8,
    lily, 0.2,
    0  # 亮度调节
)
cv2.imshow("add_w",add_w)

cv2.waitKey()
