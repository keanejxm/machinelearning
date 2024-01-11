#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  27_拟合轮廓的多边形.py
:time  2024/1/11 16:54
:desc  拟合轮廓的多边形
"""
import cv2

from 深度学习.图像处理.opencv练习.img_config import *
img = cv2.imread(f"{img_path}/cloud.png")
cv2.imshow("img",img)

# 灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)

# 二值化
t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("binary",binary)

# 查找轮廓
cnts,hie = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

adp1 = img.copy()
eps = 0.005 * cv2.arcLength(cnts[0],True)

points = cv2.approxPolyDP(cnts[0],eps,True)

adp1 = cv2.drawContours(adp1,[points],0,(0,0,255),2)
cv2.imshow("adp1",adp1)

adp2 = img.copy()
esp2 = 0.01*cv2.arcLength(cnts[0],True)
points = cv2.approxPolyDP(cnts[0],esp2,True)
adp2 = cv2.drawContours(adp2,[points],0,(0,0,255),2)
cv2.imshow("adp2",adp2)
cv2.waitKey()
cv2.destroyAllWindows()
