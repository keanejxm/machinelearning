#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:05
# @Author  : keane
# @Site    : 
# @File    : 07_二值化.py
# @Software: PyCharm

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/CPU3.png")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 二值化
t, binary = cv2.threshold(gray,  # 灰度图像
                          160,  # 阈值
                          255,  # 大于阈值转为255
                          cv2.THRESH_BINARY)  # 二值化
cv2.imshow("binary", binary)

cv2.waitKey()
