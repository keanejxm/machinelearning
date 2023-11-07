#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  02_opencv_demo.py
:time  2023/11/7 17:42
:desc  
"""
import cv2
import numpy as np

img = cv2.imread("data/lena.jpg")
cv2.imshow("img", img)

# 彩色转为灰度图像bgr---->gray
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",img_gray)

# 色彩通道操作




cv2.waitKey(2 * 1000)
cv2.destroyAllWindows()
