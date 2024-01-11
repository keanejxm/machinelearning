#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  14_透视变换.py
:time  2024/1/11 15:16
:desc  透视变换
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/pers.png")
cv2.imshow("img", img)

h, w = img.shape[:2]
pts1 = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])  # 输入图像四个顶点坐标
pts2 = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])  # 输出图像四个顶点坐标

m = cv2.getPerspectiveTransform(pts1,  # 变换之前的坐标
                                pts2)  # 变换之后的坐标
# 透视变换
res = cv2.warpPerspective(img,
                          m,
                          (w, h))

cv2.imshow("res", res)

cv2.waitKey()
