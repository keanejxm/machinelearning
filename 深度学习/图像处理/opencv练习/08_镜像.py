#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:11
# @Author  : keane
# @Site    : 
# @File    : 08_镜像.py
# @Software: PyCharm
from 深度学习.图像处理.opencv练习.img_config import *
img = cv2.imread(f"{img_path}/lena.jpg")
cv2.imshow("img",img)

# 垂直镜像0
flip_0 = cv2.flip(img,0)
cv2.imshow("flip_0",flip_0)
# 水平镜像1
flip_1 = cv2.flip(img,1)
cv2.imshow("flip_1",flip_1)


cv2.waitKey()