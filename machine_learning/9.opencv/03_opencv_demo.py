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

img = cv2.imread("data/opencv2.png")
cv2.imshow("img", img)
# 色彩通道操作
# 蓝色通道显示
b = img[:,:,0]
cv2.imshow("b",b)

img[:,:,0] = 0
cv2.imshow("b0",img)

img[:,:,1] = 0
cv2.imshow("b1",img)




cv2.waitKey(2 * 1000)
cv2.destroyAllWindows()
