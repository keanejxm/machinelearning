#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  11_opencv_erode.py
:time  2023/11/8 10:34
:desc  
"""
import cv2
import numpy as np

img = cv2.imread("data/5.png")

kernel = np.ones(shape=(3,3),dtype=np.uint8)
res = cv2.erode(img,kernel=kernel,iterations=3)
cv2.imshow("res",res)



cv2.waitKey(3*1000)
cv2.destroyAllWindows()