#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:47
# @Author  : keane
# @Site    : 
# @File    : 彩色图像直方图均衡化.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
from 深度学习.图像处理.opencv练习.img_config import img_path, opencv_data_path

img=cv2.imread(f"{img_path}/sunrise.jpg")
cv2.imshow("img",img)

# BGR---->YUV
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

# YUV----->BGR
res = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
cv2.imshow("res",res)

cv2.waitKey()