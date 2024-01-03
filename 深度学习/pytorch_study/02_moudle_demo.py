#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 19:37
# @Author  : keane
# @Site    : 
# @File    : 02_moudle_demo.py
# @Software: PyCharm
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self):
        # 调用父类方法
        super(LinearRegression, self).__init__()
        # 定义所需操作
        self.linear = nn.Linear(1, 1)

    # 重写forward方法(前向传播)
    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    # 创建一组样本数据
    data = torch.linspace(-1, 1, 40)
    x = Variable(torch.unsqueeze(data, dim=1))  # 拓展成二维
    y = Variable(x * 2 + 5 + torch.rand(x.size()))  # 计算y值

    model = LinearRegression()

    num_epochs = 1000
    lr = 0.01
    Loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        y_pred = model(x)  # 自动调用前向传播，根据输入计算输出
        loss = Loss(y_pred, y)  # 计算loss
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        optimizer.zero_grad()  # 清除梯度，防止梯度累计

        epoch_list.append(epoch)  # 记录迭代次数
        loss_list.append(torch.detach(loss).numpy())
        if epoch % 100 == 0:
            print(epoch,loss)
