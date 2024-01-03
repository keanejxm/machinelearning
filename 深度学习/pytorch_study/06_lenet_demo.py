#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 21:29
# @Author  : keane
# @Site    : 
# @File    : 06_lenet_demo.py
# @Software: PyCharm
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import os


class LetNet(nn.Module):
    def __init__(self):
        super(LetNet, self).__init__()

        # 卷积池化部分
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5)),  # 输入（3,32,32），输出（16,28,28）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 输出（16,14,14）

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5)),  # 输入（3,32,32），输出（16,28,28）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 全连接层部分
        self.classifer = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),  # fc1,输入320个特征，输出120个特征
            nn.ReLU(inplace=True),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),

            nn.Linear(84, 10),

        )

    def forward(self, x):
        features = self.conv_block(x)
        flat_features = torch.flatten(features, start_dim=1)  # 扁平化处理
        out = torch.nn.Softmax(dim=1)(self.classifer(flat_features))
        return out


if __name__ == '__main__':
    model_path = "model/letnet.pt"

    # 图像预处理
    transform = transforms.Compose([
        # 把多个操作放到一个列表里面
        transforms.ToTensor(),  # 转张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 训练集读取器
    train_set = torchvision.datasets.CIFAR10(
        root="./",
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=36,
        shuffle=True,
        num_workers=0
    )

    # 评估集读取器
    val_set = torchvision.datasets.CIFAR10(
        root="./",
        train=True,
        download=True,
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=5000,
        shuffle=False,
        num_workers=0
    )

    val_data_iter = iter(val_loader)  # 迭代器
    val_image, val_label = next(val_data_iter)  # 调用迭代器取出一批数据
    net = LetNet()  # 实例化模型
    # 判断模型是否存在，存在则加载
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("加载模型参数结束")

    # 损失函数、优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(5):
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print("[%d, %5d] loss:%.4f, val_accuracy:%.4f"
                          % (epoch, step, running_loss / 100, acc))
                    running_loss = 0
    print("训练结束")
    torch.save(net.state_dict(),model_path)