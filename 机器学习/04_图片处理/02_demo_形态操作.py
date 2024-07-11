#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_demo_.py
:time  2024/7/10 15:13
:desc  图片的形态学操作
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class DealImageForm:
    def __init__(self):
        self.image_path = "E:\keane_data\opencv_img\img_data"

    def fetch_image_path(self, image_name):
        image_path = os.path.join(self.image_path, image_name)
        return image_path

    # 图片的平移
    def translate_img(self):
        """
        平移
        :param img:
        :param x:
        :param y:
        :return:
        """

        def translate(img, x, y):
            h, w = img.shape[:2]
            # 平移矩阵
            m = np.float32([
                [1, 0, x],
                [0, 1, y]
            ])
            res = cv2.warpAffine(
                img,
                M=m,
                dsize=(w, h)  # 输出尺寸
            )
            return res

        image_path = self.fetch_image_path("lena.jpg")
        img = cv2.imread(image_path)
        cv2.imshow("img", img)
        img_t = translate(img, 20, 20)
        cv2.imshow("img_t", img_t)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图片的旋转
    def rotate_img(self):
        def rotate(img, angle, center=None):
            """
            图片的旋转
            :param img:
            :param angle: 角度
            :param center: 中心位置
            :return:
            """
            h, w = img.shape[:2]
            # 旋转矩阵
            m = cv2.getRotationMatrix2D(
                center,  # 旋转中心
                angle,  # 角度
                1.0  # 缩放比例
            )
            res = cv2.warpAffine(img, m, dsize=(w, h))
            return res

        img_path = self.fetch_image_path("lena.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        img_r = rotate(img, 45)
        cv2.imshow("img_r", img_r)

        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图片的镜像
    def flip_img(self):
        # img_path = self.fetch_image_path("lena.jpg")
        img_path = self.fetch_image_path("11111.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        # 垂直镜像 flipCode=0
        vertical = cv2.flip(img, flipCode=0)
        cv2.imshow("vertical", vertical)
        # 水平镜像
        level = cv2.flip(img, flipCode=1)
        cv2.imshow("level", level)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像缩放
    def reduce_img(self):
        img_path = self.fetch_image_path("11111.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        # 缩放
        h, w = img.shape[:2]
        reduce_res = cv2.resize(img, dsize=(int(w / 2), int(h / 2)))
        cv2.imshow("reduce_res", reduce_res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def expand_img(self):
        img_path = self.fetch_image_path("lena.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        # 扩大
        h, w = img.shape[:2]
        # 邻近插值法
        near_img = cv2.resize(img, dsize=(int(w * 2), int(h * 2)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("near_img", near_img)
        # 线性插值法
        line_img = cv2.resize(img, dsize=(int(w * 2), int(h * 2)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("line_img", line_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像剪裁
    def crop_img(self):
        # 随机剪裁
        def random_crop(img, cw, ch):
            h, w = img.shape[:2]
            start_x = np.random.randint(0, w - cw)
            start_y = np.random.randint(0, h - ch)
            res = img[start_y:start_y + ch, start_x:start_x + cw]
            return res

        # 中心剪裁
        def center_crop(img, cw, ch):
            h, w = img.shape[:2]
            start_x = int(w / 2) - int(cw / 2)
            start_y = int(h / 2) - int(ch / 2)
            res = img[start_y:start_y + ch, start_x:start_x + cw]
            return res

        img_path = self.fetch_image_path("lena.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        random_res = random_crop(img, 200, 200)
        cv2.imshow("random_res", random_res)
        center_res = center_crop(img, 200, 200)
        cv2.imshow("center_res", center_res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的相加
    def add_img(self):
        img_path_lena = self.fetch_image_path("lena.jpg")
        img_path_lily = self.fetch_image_path("lily_square.png")
        img_lena = cv2.imread(img_path_lena)
        img_lily = cv2.imread(img_path_lily)

        cv2.imshow("lena", img_lena)
        cv2.imshow("lily", img_lily)
        # 相加
        add = cv2.add(img_lena, img_lily)
        cv2.imshow("add", add)

        # 按权重相加
        add_w = cv2.addWeighted(
            img_lena, 0.8,
            img_lily, 0.2,
            gamma=0  # 亮度调节
        )
        cv2.imshow("add_w", add_w)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 透视变换
    def perspective_img(self):
        img_path = self.fetch_image_path("pers.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        h, w = img.shape[:2]
        pts1 = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])  # 输入图像四个顶点坐标
        pts2 = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])  # 输出图像四个顶点坐标

        m = cv2.getPerspectiveTransform(
            pts1,
            pts2
        )
        res = cv2.warpPerspective(
            img,
            m,
            dsize=(w, h)
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的腐蚀
    def corrosion_img(self):
        img_path = self.fetch_image_path("5.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.erode(img, kernel, iterations=3)
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像膨胀
    def swell_img(self):
        img_path = self.fetch_image_path("9.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.dilate(img, kernel, iterations=4)
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的开运算
    def open_operate_img(self):
        img_path = self.fetch_image_path("5.png")
        img = cv2.imread(img_path)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        # 开运算
        res = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel, iterations=3)
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的闭运算
    def close_operate_img(self):
        img_path = self.fetch_image_path("9.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.morphologyEx(
            img,
            op=cv2.MORPH_CLOSE,
            kernel=kernel,
            iterations=10
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的形态学梯度
    def form_grad_img(self):
        img_path = self.fetch_image_path("6.png")
        img = cv2.imread(img_path)
        cv2.imshow("img",img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.morphologyEx(
            img,
            op=cv2.MORPH_GRADIENT,
            kernel=kernel,
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的礼帽运算
    def form_hat_img(self):
        img_path = self.fetch_image_path("6.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.morphologyEx(
            img,
            op=cv2.MORPH_TOPHAT,
            kernel=kernel,
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的黑帽运算
    def blank_hat_img(self):
        img_path = self.fetch_image_path("6.png")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        res = cv2.morphologyEx(
            img,
            op=cv2.MORPH_BLACKHAT,
            kernel=kernel,
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    dif_obj = DealImageForm()
    # dif_obj.translate_img()
    # dif_obj.rotate_img()
    # dif_obj.flip_img()
    # dif_obj.reduce_img()
    # dif_obj.expand_img()
    # dif_obj.crop_img()
    # dif_obj.add_img()
    # dif_obj.perspective_img()
    # dif_obj.corrosion_img()
    # dif_obj.swell_img()
    # dif_obj.open_operate_img()
    # dif_obj.close_operate_img()
    # dif_obj.form_grad_img()
    # dif_obj.form_hat_img()
    dif_obj.blank_hat_img()