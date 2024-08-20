#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  03_demo_梯度操作.py
:time  2024/7/18 17:02
:desc  梯度操作、模糊、锐化、边沿检测、拟合轮廓
"""
import cv2
import numpy as np


class ImageGradient:
    def __init__(self):
        pass

    @staticmethod
    def read_img(img_path):
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        return img

    # 模糊：均值滤波，高斯滤波，中值滤波
    def img_dim(self):
        image_path = r"E:\keane_data\opencv_img\img_data\salt.jpg"
        img = self.read_img(image_path)
        # 均值滤波
        ave_res = cv2.blur(img, (3, 3))
        # 高斯滤波
        gaussian_res = cv2.GaussianBlur(
            img,
            (3, 3)
            , 1  # 标准差
        )
        # 中值滤波
        median_res = cv2.medianBlur(
            img,
            5
        )
        # 自定义卷积核
        filter_w = np.ones(shape=(5, 5)) / 25
        res = cv2.filter2D(img, -1, kernel=filter_w)
        cv2.imshow("ave_res", ave_res)
        cv2.imshow("gaussian_res", gaussian_res)
        cv2.imshow("median_res", median_res)
        cv2.imshow("res", res)

        cv2.waitKey()
        cv2.destroyAllWindows()

    # 图形的锐化:边沿检测
    def img_sharpen(self):
        """
        边沿检测：sobel算子、laplacian算子、canny算子
        :return:
        """
        img_path = r"E:\keane_data\opencv_img\img_data\lily.png"
        img = self.read_img(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        cv2.imshow("sobel", sobel)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        cv2.imshow("lap", lap)
        canny = cv2.Canny(gray, 50, 127)
        cv2.imshow("canny", canny)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 查找轮廓

    def find_outline(self, img_path):
        """"""
        img = self.read_img(img_path)
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化
        t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("binary", binary)
        # 查找轮廓
        cnts, hie = cv2.findContours(
            binary,
            mode=cv2.RETR_EXTERNAL,  # 只检查外部轮廓
            method=cv2.CHAIN_APPROX_NONE,  # 保存所有坐标点
        )
        # cnts:三维坐标点的数组，hie:轮廓的位置信息
        return img, cnts, hie

    # 绘制轮廓
    def draw_outline(self):
        """"""
        img_path = r"E:\keane_data\opencv_img\img_data\3.png"
        img, cnts, hie = self.find_outline(img_path)
        # 绘制轮廓
        res = cv2.drawContours(
            img,
            contours=cnts,
            contourIdx=-1,
            color=(0, 0, 255),
            thickness=2,  # 线条粗细
        )
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 拟合轮廓---矩形
    def fitting_outline_rect(self):
        """
        矩形、圆形、多边形
        :return:
        """
        img_path = r"E:\keane_data\opencv_img\img_data\cloud.png"
        img, cnts, bie = self.find_outline(img_path)
        # --------------------------------矩形------------------------------------
        x, y, dw, dh = cv2.boundingRect(cnts[0])  # 返回值为x,y坐标以及宽和高
        # 生成绘制轮廓点的坐标
        points = np.array(
            [
                [[x, y]],
                [[x, y + dh]],
                [[x + dw, y + dh]],
                [[x + dw, y]]
            ]
        )
        res = cv2.drawContours(img, [points], -1, (0, 0, 255))
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 拟合轮廓---圆形
    def fitting_outline_circle(self):
        """
        矩形、圆形、多边形
        :return:
        """
        img_path = r"E:\keane_data\opencv_img\img_data\cloud.png"
        img, cnts, bie = self.find_outline(img_path)
        # 生成绘制轮廓点的坐标
        center, radius = cv2.minEnclosingCircle(cnts[0])
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        res = cv2.circle(img, center, radius, color=(0, 0, 255), thickness=2)
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 拟合轮廓---多边形
    def fitting_outline_polygon(self):
        """
        矩形、圆形、多边形
        :return:
        """
        img_path = r"E:\keane_data\opencv_img\img_data\cloud.png"
        img, cnts, bie = self.find_outline(img_path)
        # 生成绘制轮廓点的坐标
        eps = 0.005 * cv2.arcLength(cnts[0], True)
        points = cv2.approxPolyDP(cnts[0], epsilon=eps, closed=True)
        res = cv2.drawContours(img, [points], 0, (0, 0, 255), 2)
        cv2.imshow("res", res)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ig_obj = ImageGradient()
    # ig_obj.img_dim()
    # ig_obj.img_sharpen()
    # 绘制轮廓
    # ig_obj.draw_outline()
    # 拟合轮廓---矩形
    # ig_obj.fitting_outline_rect()
    # 拟合轮廓--圆形
    # ig_obj.fitting_outline_circle()
    # 拟合轮廓-多边形
    # ig_obj.fitting_outline_polygon()
