#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  22_paper.py
:time  2024/1/11 15:57
:desc  
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/paper.jpg")

cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 二值化
t, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

canny = cv2.Canny(gaussian, 30, 120)

cv2.imshow("canny", canny)

cv2.waitKey()
