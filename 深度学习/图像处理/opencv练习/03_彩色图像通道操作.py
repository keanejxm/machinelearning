#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:19
# @Author  : keane
# @Site    : 
# @File    : 彩色图像通道操作.py
# @Software: PyCharm
import cv2
from 深度学习.图像处理.opencv练习.img_config import img_path, opencv_data_path

img = cv2.imread(f"{img_path}/opencv2.png")

cv2.imshow("img", img)
# 取出蓝色通道并显示shape= （高宽通道）（rgb）
b = img[:, :, 2]
cv2.imshow("b", b)
# 在原始图像上，将蓝色通道赋值为0
img[:,:,0] = 0
cv2.imshow("b0",img)
# 在蓝色通道为0的基础身上，再将绿色通道设置为0
img[:,:,1] =0
cv2.imshow("b0_g0",img)


cv2.waitKey()
