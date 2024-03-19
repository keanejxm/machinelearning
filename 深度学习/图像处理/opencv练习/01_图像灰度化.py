#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 7:40
# @Author  : keane
# @Site    : 
# @File    : 图像灰度化.py
# @Software: PyCharm

import cv2
from 深度学习.图像处理.opencv练习.img_config import img_path, opencv_data_path

img = cv2.imread(f"{img_path}/lena.jpg", flags=1)  # 1:彩色 0：灰度
cv2.imshow("img", img)

# 彩色---->灰度  BGR--->GRAY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)

cv2.waitKey()
cv2.destroyAllWindows()
