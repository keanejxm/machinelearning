#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  13_图像的相减.py
:time  2024/1/11 15:12
:desc  对应位置对应相减
"""
from 深度学习.图像处理.opencv练习.img_config import *

img3 = cv2.imread(f"{img_path}/3.png",0)
img4 = cv2.imread(f"{img_path}/4.png",0)

cv2.imshow("img3",img3)
cv2.imshow("img4",img4)

# 减法
res = cv2.subtract(img3,img4)
cv2.imshow("res",res)

cv2.waitKey()