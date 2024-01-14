#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  21_边缘检测.py
:time  2024/1/11 15:48
:desc  边缘检测
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/lily.png", 0)

cv2.imshow("img", img)

# sobel梯度算子 一阶梯度
sobel = cv2.Sobel(img,
                  cv2.CV_64F,
                  1,#水平方向上的到导数
                  1,# 数值方向的导数1:1阶导数
                  ksize=5)
cv2.imshow("sobel", sobel)

# laplacian梯度算子
lap = cv2.Laplacian(img,
                    cv2.CV_64F,
                    ksize=5)
cv2.imshow("lap", lap)

# canny
canny = cv2.Canny(img, 50, 220)
cv2.imshow("canny", canny)

cv2.waitKey()
