#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_demo.py
:time  2023/11/7 17:22
:desc  opencv
"""
import cv2

img = cv2.imread("data/Linus.png")
print(img.shape)
print(img)
# 显示
cv2.imshow("img",img)

# 主动进入阻塞状态，等待用户按下某个按键
# cv2.waitKey(delay=1000*2 )
cv2.waitKey()
# print(a)
# cv2.destroyAllWindows()


# 处理图像

# 保存图像
cv2.imwrite("new_img.png",img)
