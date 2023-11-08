#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  13_opencv_morphologyex.py
:time  2023/11/8 10:42
:desc  
"""
import cv2
import numpy as np

img = cv2.imread("data/5.png")
kernel = np.ones(shape=(3, 3), dtype=np.uint8)
# 开运算
res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel,iterations=4)
cv2.imshow("res", res)

# 闭运算
res1 = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel=kernel,iterations=4)
cv2.imshow("res1", res1)

# 图像形态学梯度
res1 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel=kernel,iterations=4)
cv2.imshow("res1", res1)

cv2.waitKey(3 * 1000)
cv2.destroyAllWindows()
