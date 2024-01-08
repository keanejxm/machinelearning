#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 22:05
# @Author  : keane
# @Site    : 
# @File    : 练习1.py
# @Software: PyCharm
import cv2
import os
from common_utils import DATA_PATH


opencv_data_path = f"{DATA_PATH}/opencv_img"
img_path = f"{DATA_PATH}/opencv_img/img_data"



# 加载图像
img = cv2.imread(os.path.join(img_path,"Linus.png"))
# 显示
cv2.imshow("img",img)
# print(img)
cv2.imshow("img2",img)

# 保存图像
cv2.imwrite(f"{opencv_data_path}/new_Linus.png",img)

# 阻塞
cv2.waitKey()
cv2.destroyAllWindows()
