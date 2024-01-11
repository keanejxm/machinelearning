#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  25_拟合轮廓的矩形包围框.py
:time  2024/1/11 16:36
:desc  拟合轮廓的矩形包围框
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/cloud.png")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, code=cv2.COLOR_BGRA2GRAY)
cv2.imshow("gray", gray)
# 二值化
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# 查找轮廓
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# 根据轮廓点坐标，生成拟合轮廓的坐标信息
x, y, w, h = cv2.boundingRect(cnts[0])

points = np.array(
    [[[x, y]],  # 左上角
     [[x, y + h]],  # 左下角
     [[x + w, y + h]],  # 右下角
     [[x + w, y]]]  # 右上角
)

res = cv2.drawContours(img,[points],0,(0,0,255),2)
cv2.imshow("res",res)

cv2.waitKey()
cv2.destroyAllWindows()