#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  08_opencv_demo.py
:time  2023/11/8 10:14
:desc  
"""
import cv2
import numpy as np

# 图像相加

img1 = cv2.imread("data/lena.jpg", 0)
img2 = cv2.imread("data/lily_square.png", 0)

add_res = cv2.add(img1, img2)
cv2.imshow("res", add_res)

# 按照权重相加
add_res1 = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)
cv2.imshow("res1", add_res1)





cv2.waitKey(delay=3 * 1000)
cv2.destroyAllWindows()
