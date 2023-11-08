#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  15_opencv_edge.py
:time  2023/11/8 14:30
:desc  
"""
import cv2

img = cv2.imread("data/lily.png",0)

sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
cv2.imshow("sobel",sobel)



canny = cv2.Canny(img,50,200)
cv2.imshow("canny",canny)


cv2.waitKey(3*1000)
cv2.destroyAllWindows()