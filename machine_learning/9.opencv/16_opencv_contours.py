#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  16_opencv_contours.py
:time  2023/11/8 15:51
:desc  查找轮廓
"""
import cv2

img = cv2.imread("data/3.png")
cv2.imshow("img", img)

# 灰度话
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
# 二值化
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)
# 查找轮廓
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# for cnt in cnts:
#     print(cnt.shape)
# print(hie)

# 绘制轮廓
res= cv2.drawContours(img,
                 contours=cnts,
                 contourIdx=-1,
                 color=(0, 0, 255),
                 thickness=2  # 线条粗细
                 )

cv2.imshow("res",res)

cv2.waitKey(4 * 1000)
cv2.destroyAllWindows()
