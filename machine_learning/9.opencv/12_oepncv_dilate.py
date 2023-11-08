#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  12_oepncv_dilate.py
:time  2023/11/8 10:40
:desc  
"""
import cv2
import numpy as np

img = cv2.imread("data/9.png")

kernel = np.ones(shape=(3,3),dtype=np.uint8)
res = cv2.dilate(img,kernel=kernel,iterations=4)
cv2.imshow("res",res)



cv2.waitKey(3*1000)
cv2.destroyAllWindows()