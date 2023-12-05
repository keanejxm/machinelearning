#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  mult_demo_02.py
:time  2023/12/5 15:43
:desc  两个张量相加，相乘
"""
import numpy as np
import paddle.fluid as fluid

# 定义两个张量
x = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")

x_add_y = fluid.layers.elementwise_add(x, y)
x_mul_y = fluid.layers.elementwise_mul(x, y)

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 初始化网络
exe.run(program=fluid.default_startup_program())
# 数据参数
params = {
    "x": np.array([[1, 2, 3], [4, 5, 6]]),
    "y": np.array([[1, 1, 1], [2, 2, 2]])
}
outs = exe.run(program=fluid.default_main_program(),
               feed=params,
               fetch_list=[x_add_y, x_mul_y])
print(outs[0])
print(outs[1])
