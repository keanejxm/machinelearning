#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:24
# @Author  : keane
# @Site    : 
# @File    : 10_图像缩放.py
# @Software: PyCharm
from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/Linus.png")

cv2.imshow("img",img)

# 缩小
h,w = img.shape[:2]
dst_size = (int(w/2),int(h/2))
reduce = cv2.resize(img,dst_size)
cv2.imshow("reduce",reduce)

# 放大
dst_size = (w*2,h*2)
# 最邻近插值法
near = cv2.resize(img,
                  dst_size,
                  interpolation=cv2.INTER_NEAREST)
cv2.imshow("near",near)

# 双线性插值法
linear = cv2.resize(img,
                    dst_size,
                    interpolation=cv2.INTER_LINEAR)
cv2.imshow('linear',linear)

cv2.waitKey()