#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  23_查找并绘制轮廓.py
:time  2024/1/11 16:03
:desc  查找并绘制轮廓
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/3.png")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow("gray", gray)

# 二值化
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# 查找轮廓
cnts, hie = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,  # 只检查外层轮廓
    cv2.CHAIN_APPROX_NONE,  # 保存所有坐标
)

for cnt in cnts:
    print(cnt.shape)
print(hie)

# 绘制轮廓
res = cv2.drawContours(img,  # 在那张图像上画
                       cnts,  # 所有的轮廓坐标
                       -1,  # 要绘制的轮廓索引
                       (0, 0, 255),  # 颜色 BGR
                       -1)  # 线条粗细 px
cv2.imshow("res", res)

cv2.waitKey()
