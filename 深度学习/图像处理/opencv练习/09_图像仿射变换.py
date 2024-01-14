#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:14
# @Author  : keane
# @Site    : 图像仿射变换：平移、旋转
# @File    : 09_图像仿射变换.py
# @Software: PyCharm
from 深度学习.图像处理.opencv练习.img_config import *


def translate(img, x, y):
    """
    平移
    Args:
        img:
        x:
        y:

    Returns:

    """
    h, w = img.shape[:2]  # [h,w,c]

    m = np.float32([[1, 0, x],
                    [0, 1, y]])
    res = cv2.warpAffine(img,
                         m,
                         (w, h))# 输出尺寸
    return res


def rotate(img, angle, center=None):
    """
    旋转
    Args:
        img:
        angle:
        center:

    Returns:

    """
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # 旋转矩阵
    m = cv2.getRotationMatrix2D(center,  # 旋转中心
                                angle,  # 旋转角度
                                1.0)  # 缩放比例
    res = cv2.warpAffine(img,
                         m,
                         (w, h))
    return res


if __name__ == '__main__':
    img = cv2.imread(f"{img_path}/lena.jpg")
    cv2.imshow("img", img)
    # 平移
    translated = translate(img, 50, 50)
    cv2.imshow("translated", translated)

    # 旋转
    rotated = rotate(img, 45)
    cv2.imshow("rotated", rotated)

    cv2.waitKey()
