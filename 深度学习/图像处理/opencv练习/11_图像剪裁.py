 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 22:42
# @Author  : keane
# @Site    : 
# @File    : 图像剪裁.py
# @Software: PyCharm
from 深度学习.图像处理.opencv练习.img_config import *


# 随机剪裁
def random_crop(img, cw, ch):
    h, w = img.shape[:2]
    start_x = np.random.randint(0, w - cw)
    start_y = np.random.randint(0, h - ch)
    # (h,w,c)
    res = img[start_y:start_y + ch, start_x:start_x + cw]
    return res


# 中心剪裁
def center_crop(img, cw, ch):
    h, w = img.shape[:2]
    start_x = int(w / 2) - int(cw / 2)
    start_y = int(h / 2) - int(ch / 2)

    # (h,w,c)
    res = img[start_y:start_y + ch, start_x:start_x + cw]
    return res


if __name__ == '__main__':
    img = cv2.imread(f"{img_path}/banana_1.png")
    cv2.imshow("img", img)

    # 随机剪裁
    random_res = random_crop(img, 200, 200)
    cv2.imshow("random", random_res)

    # 中心剪裁
    center_res = center_crop(img, 200, 200)
    cv2.imshow("center_res", center_res)

    cv2.waitKey()
