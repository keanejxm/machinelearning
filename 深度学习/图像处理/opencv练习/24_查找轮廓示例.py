#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  24_查找轮廓示例.py
:time  2024/1/11 16:10
:desc  cpu示例
"""

from 深度学习.图像处理.opencv练习.img_config import *

img = cv2.imread(f"{img_path}/CPU3.png")
cv2.imshow("img", img)

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow("gray", gray)

# 二值化
t, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", binary)

# 查找轮廓
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

# 构建一张纯黑色的图像，shape = 二值化图像,mask

mask = np.zeros_like(binary)
cv2.imshow("mask", mask)

# 将查找的度盘的轮廓画在mask图像上，使用实心填充（线条粗细为-1）
img_fill = cv2.drawContours(mask, cnts, -1, 255, -1)
cv2.imshow("img_fill", img_fill)

# 让二值化的图像，和画完轮廓的图像，做减法
img_sub = cv2.subtract(img_fill, binary)
cv2.imshow("img_sub", img_sub)

# 对瑕疵做闭运算（先膨胀再腐蚀）
img_close = cv2.morphologyEx(
    img_sub,
    cv2.MORPH_CLOSE,
    (5, 5),
    iterations=2
)
cv2.imshow("img_close", img_close)

# 查找面积最大的瑕疵轮廓

cnts, hies = cv2.findContours(img_close,
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)

# 计算每个轮廓的面积，按照面积进行排序
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,
                  reverse=True)
# 拟合最大瑕疵的轮廓的最小外接圆（行业标准为8，瑕疵个数不能超过3个）
area = cv2.contourArea(cnts[0])
print(area)

if (area > 8) or (len(cnts) >= 3):
    center, radius = cv2.minEnclosingCircle(cnts[0])
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    # 将瑕疵的最小外接圆，画在img原始图像上
    print("有瑕疵，瑕疵面积为:", area)
    res = cv2.circle(img, center, radius, (0, 0, 255), 2)
    cv2.imshow("res", res)
cv2.waitKey()
cv2.destroyAllWindows()
