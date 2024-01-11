#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  20_图像的模糊.py
:time  2024/1/11 15:41
:desc  均值滤波/高斯滤波、中值滤波
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/salt.jpg")
cv2.imshow("img", img)

# 均值滤波
blur = cv2.blur(img, (5, 5))
cv2.imshow("blur", blur)

# 高斯滤波
gaussian = cv2.GaussianBlur(img, (5, 5), 1)
cv2.imshow("gaussian", gaussian)

# 中值滤波
median = cv2.medianBlur(img, 5)
cv2.imshow("median", median)
cv2.imwrite(f"{opencv_data_path}/qqq.jpg", median)

# 自定义卷积核，调用卷积
filter_w = np.ones(shape=(5, 5)) / 25.0

res = cv2.filter2D(img,
                   -1,  # 图像的深度，-1代表和原图一致
                   filter_w)

cv2.imshow("res",res)


cv2.waitKey()
