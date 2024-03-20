#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:55
# @Author  : keane
# @Site    : 
# @File    : 06_提取彩色图像中指定的颜色.py
# @Software: PyCharm

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/opencv2.png")
cv2.imshow("img", img)

# BGR---->HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

min_val = np.array([0, 50, 50])
max_val = np.array([10, 255, 255])

mask = cv2.inRange(hsv, min_val, max_val)

cv2.imshow("mask", mask)

# 让原始图像mask和原始图像mask做位与计算
res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("res", res)

cv2.waitKey()
