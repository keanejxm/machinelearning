#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  image_practice.py
:time  2024/3/18 15:31
:desc  图像练习
"""
import os
import cv2
import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.fluid as fluid
from common_utils import DATA_PATH

cv_image_path = os.path.join(DATA_PATH, "opencv_img")
fruits_path = os.path.join(DATA_PATH, "fruits")

image_path = os.path.join(cv_image_path, "img_data")

image_path_paper = os.path.join(image_path, "paper.jpg")

img = cv2.imread(image_path_paper,flags=0)

cv2.imshow("img",img)

t,binary = cv2.threshold(src=img,thresh=200,maxval=255,type=cv2.THRESH_BINARY)
cv2.imshow("binary",binary)

canny = cv2.Canny(image=img,threshold1=30,threshold2=120)
cv2.imshow("canny",canny)

cv2.waitKey()
cv2.destroyAllWindows()
