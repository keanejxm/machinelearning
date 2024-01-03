#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 20:25
# @Author  : keane
# @Site    : 
# @File    : 04_sequential.py
# @Software: PyCharm

import torch
from torch import nn


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # 通过Sequential定义多个操作层
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5)),  # 输入通道、输出通道、卷积核大小
            nn.ReLU(),  # 激活层
            nn.Conv2d(20, 64, (5, 5)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x


if __name__ == '__main__':
    net = Net2()
    print(net)
    for param in net.parameters():  # 打印所有参数
        print(type(param.data), param.size())
