#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_demo_.py
:time  2024/7/10 15:13
:desc
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class DealImage:
    def __init__(self):
        self.image_path = "E:\keane_data\opencv_img\img_data"

    def fetch_image_path(self, image_name):
        image_path = os.path.join(self.image_path, image_name)
        return image_path

    # 图像灰度化
    def gray_image(self):
        image_path = self.fetch_image_path("lena.jpg")
        img = cv2.imread(image_path)
        # 图像灰度化--->方法1
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_1 = cv2.imread(image_path, flags=0)  # 0->灰色 1->彩色
        cv2.imshow("gray_img_1", gray_img_1)
        cv2.imshow("gray_img", gray_img)
        cv2.imshow("img", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图像的色彩通道操作
    def color_channel(self):
        image_path = self.fetch_image_path("opencv.png")
        img = cv2.imread(image_path)
        # img(高宽通道)(300, 300, 3) BGR(颜色通道顺序) 0-黑 255-白
        height, width, channel_num = img.shape
        blue = img[:, :, 0]  # 单通道-灰度图
        cv2.imshow("img", img)
        cv2.imshow("blue", blue)
        # 将蓝色通道变为黑色
        img[:, :, 0] = 0
        cv2.imshow("blue_blank", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 灰度图直方图化
    def gray_histogram(self):
        """"""
        img_path = self.fetch_image_path("sunrise.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray_img", gray_img)
        # 灰度图直方图化
        gray_equal = cv2.equalizeHist(gray_img)
        cv2.imshow("gray_equal", gray_equal)

        # matplotlib展示直方图
        plt.figure("img_hist")
        plt.subplot(2, 1, 1)
        plt.hist(
            img.ravel(),
            bins=256
        )
        plt.subplot(2, 1, 2)
        plt.hist(
            gray_equal.ravel(),
            bins=256
        )
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 彩色图像直方图均衡化
    def color_histogram(self):
        """"""
        img_path = self.fetch_image_path("sunrise.jpg")
        img = cv2.imread(img_path)
        cv2.imshow("img", img)

        # 彩色图像直方图均衡化先将BGR转为YUV,然后将亮度均衡化,Y-->亮度
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
        # 将YUV转为BGR
        bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        cv2.imshow("bgr_img", bgr_img)

        cv2.waitKey()
        cv2.destroyAllWindows()

    # 提取彩色图像中指定的颜色
    def extract_color(self):
        """"""
        img_path = self.fetch_image_path("opencv2.png")
        img = cv2.imread(img_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        min_val = np.array([0, 50, 50])
        max_val = np.array([10, 255, 255])
        mask = cv2.inRange(hsv_img, min_val, max_val)
        cv2.imshow("mask", mask)

        # 让原始图像mask和原始图像做位于计算
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("res", res)

        cv2.waitKey()
        cv2.destroyAllWindows()

    # 二值化
    def binary_image(self):
        """
        图像二值化
        :return:
        """
        image_path = self.fetch_image_path("CPU3.png")
        img = cv2.imread(image_path)
        cv2.imshow("img", img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray_img", gray_img)
        # 二值化
        t, binary = cv2.threshold(
            gray_img,
            160,  # 阈值
            255,  # 大于阈值的转为255
            cv2.THRESH_BINARY,  # 二值化
        )
        cv2.imshow("binary", binary)

        # 反二值化
        t,binary = cv2.threshold(
            gray_img,
            160,
            255,
            cv2.THRESH_BINARY_INV
        )
        cv2.imshow("binary_inv",binary)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    di_obj = DealImage()
    # di_obj.gray_image()
    # di_obj.color_channel()
    # di_obj.gray_histogram()
    # di_obj.color_histogram()
    # di_obj.extract_color()
    di_obj.binary_image()