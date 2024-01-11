#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  28_图像矫正.py
:time  2024/1/11 17:04
:desc  图像矫正
"""
import math

import cv2
import numpy as np

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/paper.jpg")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 二值化
# t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv2.imshow("binary",binary)
# 二值化，效果不理想，尝试使用边沿检测
# sobel = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
# cv2.imshow("sobel",sobel)
# lap = cv2.Laplacian(gray,cv2.CV_64F,ksize=5)
# cv2.imshow("lap",lap)
# 高斯滤波
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
close = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, (5, 5))
canny = cv2.Canny(close, 30, 120)
cv2.imshow("canny", canny)

# 查找轮廓
cnts, hie = cv2.findContours(canny,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# 把轮廓按照面积进行排序
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,
                  reverse=True)
# 绘制轮廓
res = cv2.drawContours(img, cnts, 0, (0, 0, 255), 2)
cv2.imshow("res", res)

# 拟合目标轮廓的多边形
eps = 0.01 * cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], eps, True)

# 绘制角点
img_copy = img.copy()
for peak in approx:
    peak = tuple(peak[0])
    cv2.circle(img_copy, peak, 10, (0, 0, 255), 2)
cv2.imshow("img_copy", img_copy)

# 变换之前的坐标点，左上角，左下角，右下角，右上角
src = approx.reshape(4, 2).astype("float32")

# 求出值的宽度和高度
h = int(math.sqrt((src[0][0] - src[1][0]) ** 2 + (src[0][1] - src[1][1]) ** 2))
w = int(math.sqrt((src[0][0] - src[3][0]) ** 2 + (src[0][1] - src[3][1]) ** 2))
# 变换之后的坐标
dst = np.array([[0, 0],  # 左上角
                [0, h],  # 左下角
                [w, h],  # 右下角
                [w, 0],  # 右上角
                ], dtype="float32")

# 根据变换之前的坐标点和变换之后的坐标点，构建透视变换的矩阵
m = cv2.getPerspectiveTransform(src, dst)

# 执行透视变换
res = cv2.warpPerspective(img, m, (w, h))
cv2.imshow("res", res)
cv2.waitKey()
cv2.destroyAllWindows()
