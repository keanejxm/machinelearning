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
import matplotlib.pyplot as plt

img = cv2.imread("data/sunrise.jpg",0)
cv2.imshow("img", img)
# 灰度图像直方图均衡化
plt.subplot(2,1,2)
plt.hist(img.ravel(),bins=256)
plt.show()

res = cv2.equalizeHist(img)
cv2.imshow("res",res)

plt.subplot(2,1,2)
plt.hist(res.ravel(),bins=256)
plt.show()



cv2.waitKey(2 * 1000)
cv2.destroyAllWindows()
