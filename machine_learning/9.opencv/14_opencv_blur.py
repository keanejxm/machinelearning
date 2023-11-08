#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  14_opencv_blur.py
:time  2023/11/8 11:17
:desc  图像模糊
"""

import cv2
import numpy as np

img = cv2.imread("data/salt.jpg")

# 均值滤波
blur_ = cv2.blur(img, ksize=(5, 5))

cv2.imshow("blur", blur_)

# 高斯滤波
gaussian = cv2.GaussianBlur(img, ksize=(5, 5),sigmaX=1)
cv2.imshow("gaussian",gaussian)

# 中值滤波
median = cv2.medianBlur(img,5)
cv2.imshow("median",median)

# 自定义卷积核
filter_w = np.ones(shape=(5,5))/25.0
res  =cv2.filter2D(img,
             -1,
             filter_w)
cv2.imshow("res",res)

cv2.waitKey(3 * 1000)
cv2.destroyAllWindows()
