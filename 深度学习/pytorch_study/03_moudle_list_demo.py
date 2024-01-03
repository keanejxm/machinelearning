#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 20:15
# @Author  : keane
# @Site    : 
# @File    : 03_moudle_list_demo.py
# @Software: PyCharm
import torch
from torch import nn


# moduleList:一旦操作存入容器，自动注册模型参数（自动参与微分与梯度下降，顺序没有要求）

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 创建两个线性层，存入ModuleList中
        self.linears = nn.ModuleList(
            [nn.Linear(10, 10) for i in range(2)]
        )

    # 重写forword方法
    def forward(self, x):
        out1 = self.linears[0](x)  # 先执行moduleList中的第一个操作
        out2 = self.linears[1](out1)  # 在执行moduleList中的第二个操作
        return out2


if __name__ == '__main__':
    net = Net1()
    print(net)
    for param in net.parameters():  # 打印所有参数
        print(type(param.data), param.size())
