#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 21:03
# @Author  : keane
# @Site    : 
# @File    : 05_module_save_demo.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    # 创建样本
    x = torch.linspace(0,1,10).reshape(10,1)
    y = x*x -0.5*x +1.5625
    net = Net()
    optimizer = optim.SGD(net.parameters(),lr =0.1)

    # 训练
    for n in range(0,10000):
        y_pred = net(x)
        loss = sum(((y_pred-y)**2)/2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if n%1000==0:
            print(n,torch.detach(loss).numpy())
    # 保存模型
    torch.save(net,"model/mynet.pt")
    print("save model ok")

    # 加载模型
    net2 = torch.load("model/mynet.pt")

    import matplotlib.pyplot as plt
    plt.plot(x,y,"k*") # 绘制样本的分布图
    test_y = []
    test_x = torch.linspace(0,1,100).reshape(100,1)
    for sample in test_x:
        y_hat = net2(sample)
        test_y.append(torch.detach(y_hat).numpy()[0]) # 取出计算的y值

    plt.plot(torch.detach(test_x).numpy(),test_y,"b-")
    plt.show()