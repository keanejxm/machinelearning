#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  26_拟合轮廓的最小圆形包围框.py
:time  2024/1/11 16:47
:desc  拟合轮廓最小的圆形包围框
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/cloud.png")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow("gray", gray)

# 二值化
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# 寻找轮廓
cnts, his = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

# 根据轮廓坐标点，生成拟合轮廓的坐标信息
center, radius = cv2.minEnclosingCircle(cnts[0])

center = (int(center[0]), int(center[1]))
radius = int(radius)

print(f"圆心：{center},半径{radius}")
res = cv2.circle(img, center, radius, (0, 0, 255), 2)

cv2.imshow("res", res)
cv2.waitKey()
cv2.destroyAllWindows()
