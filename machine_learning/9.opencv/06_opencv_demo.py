#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  06_opencv_demo.py
:time  2023/11/8 9:33
:desc  图像形态
"""
import cv2
import numpy as np

img = cv2.imread("data/lena.jpg")

# 图像镜像
flip0 = cv2.flip(img, 0)
cv2.imshow("flip0", flip0)

flip1 = cv2.flip(img, 1)
cv2.imshow("flip1", flip1)


# 仿射变换
# 平移
def translate(img, x, y):
    h, w = img.shape[:2]  # [h,w,c]
    m = np.float32([[1, 0, x], [0, 1, y]])
    res = cv2.warpAffine(img, M=m, dsize=(w, h))
    return res


# 旋转
def rotate(img, angle, center=None):
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    # 旋转矩阵
    m = cv2.getRotationMatrix2D(center=center,  # 旋转中心
                                angle=angle,  # 旋转角度
                                scale=1.0)  # 缩放比例
    res = cv2.warpAffine(img, M=m, dsize=(w, h))
    return res


# 平移
trans_res = translate(img, 50, 50)
cv2.imshow("trans", trans_res)

# 旋转
rotated = rotate(img,45)
cv2.imshow("rotated",rotated)

# 缩放
h,w = img.shape[:2]
dst_size = (int(w/2),int(h/2))
reduce = cv2.resize(img,dsize=dst_size)
cv2.imshow("reduce",reduce)

# 放大
dst_size = [w*2,h*2]
near = cv2.resize(img,dsize=dst_size,interpolation=cv2.INTER_NEAREST)
cv2.imshow("near",near)

linear = cv2.resize(img,dsize=dst_size,interpolation=cv2.INTER_LINEAR)
cv2.imshow("linear",linear)





cv2.waitKey(delay=2 * 1000)
cv2.destroyAllWindows()





