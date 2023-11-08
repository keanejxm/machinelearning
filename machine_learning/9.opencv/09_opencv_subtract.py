#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  09_opencv_subtract.py
:time  2023/11/8 10:21
:desc  图像相减
"""
import cv2
import numpy as np

img1 = cv2.imread("data/3.png",0)
img2 = cv2.imread("data/4.png",0)
res = cv2.subtract(img1,img2)
cv2.imshow("res",res)

cv2.waitKey(3*1000)
cv2.destroyAllWindows()

