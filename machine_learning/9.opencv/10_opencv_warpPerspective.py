#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  10_opencv_warpPerspective.py
:time  2023/11/8 10:25
:desc  
"""
import cv2
import numpy as np

img = cv2.imread("data/pers.png")
# 透视变换
pst1 = np.float32()
pst2 = np.float32()
h,w = img.shape[:2]
m = cv2.getPerspectiveTransform(pst1,pst2)
cv2.warpPerspective(img,
                    M=m,
                    dsize=(w,h))


cv2.waitKey(3*1000)
cv2.destroyAllWindows()