#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:31
# @Author  : keane
# @Site    : 
# @File    : 灰度图像直方图均衡化.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
from 深度学习.图像处理.opencv练习.img_config import img_path, opencv_data_path

img = cv2.imread(f"{img_path}/sunrise.jpg", 0)
cv2.imshow("img", img)

# 直方图均衡化
res = cv2.equalizeHist(img)
cv2.imshow("res",res)

# a = img.ravel()
# 直方图
# res.ravel()将多维数组转为1维数组
plt.subplot(2,2,1)
plt.hist(img.ravel(),
         bins=256,
         range=[0,256])
plt.subplot(2,1,2)
plt.hist(res.ravel(),
         bins=256,
         range = [0,2556])
plt.show()
cv2.waitKey()
