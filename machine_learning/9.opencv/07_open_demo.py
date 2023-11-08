#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  07_open_demo.py
:time  2023/11/8 9:56
:desc
"""
import cv2
import numpy as np

img = cv2.imread("data/banana_1.png")


# 随机裁剪
def random_crop(img, cw, ch):
    h, w = img.shape[:2]
    start_x = np.random.randint(0, w - cw, )
    start_y = np.random.randint(0, h - ch)
    res = img[start_y:start_y + ch, start_x:start_x + cw]
    return res


# 中心裁剪
def center_crop(img, cw, ch):
    h, w = img.shape[:2]
    center_x = int(w / 2)
    center_y = int(h / 2)
    x_start = center_x - int(cw / 2)
    x_end = center_x + int(cw / 2)

    y_start = center_y - int(ch / 2)
    y_end = center_y + int(ch / 2)
    res = img[x_start:x_end, y_start:y_end]
    return res


res_random = random_crop(img, 200, 200)
cv2.imshow("random", res_random)

res_center = center_crop(img, 200, 200)
cv2.imshow("center", res_center)

cv2.waitKey(4 * 1000)
cv2.destroyAllWindows()
