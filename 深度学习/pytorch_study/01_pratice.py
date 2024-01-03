#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/6 20:34
# @Author  : keane
# @Site    : 
# @File    : 01_pratic.py
# @Software: PyCharm


import torch
import numpy as np

a = torch.tensor(5)
print(a)

# 定义数组
arr = np.array([4])
a = torch.tensor(arr)
print(a)

# 根据指定的形状生成张量
ts_1 = torch.Tensor(5)
print(ts_1)

ts_2 = torch.Tensor(2,3)
print(ts_2)

ts_3 = torch.Tensor(2,3,4)
print(ts_3)

a = torch.Tensor([1,3])
print(a)

