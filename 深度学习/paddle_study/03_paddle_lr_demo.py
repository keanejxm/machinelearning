#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  paddle_lr_demo_03.py
:time  2023/12/5 15:53
:desc  
"""
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]]).astype("float32")
y_true = np.array([[5.0], [5.5], [6.0], [6.8], [6.8]]).astype("float32")

# 定义占位符
x = fluid.layers.data(name="x", shape=[1], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")

# 全连接神经网络
y_predict = fluid.layers.fc(input=x,
                            size=1,
                            act=None)

# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict,
                                      label=y)
avg_cost = fluid.layers.mean(cost)

# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(avg_cost)

# 执行
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())

costs = []  # 损失值
iters = []  # 迭代次数
values = []

#
params = {"x": train_data, "y": y_true}
for i in range(200):
    outs = exe.run(program=fluid.default_main_program(),
                   feed=params,
                   fetch_list=[y_predict, avg_cost])
    costs.append(outs[1])
    iters.append(i)
    print(f"第{i}次，cost:{outs[1][0]}")

# 线性模型可视化
tmp = np.random.rand(10, 1)
tmp = tmp * 2
tmp.sort(axis=0)
x_test = np.array(tmp).astype("float32")
params = {"x": x_test, "y": x_test}
y_out = exe.run(program=fluid.default_main_program(),
                feed=params,
                fetch_list=[y_predict])

# 损失函数可视化
plt.figure("Training")
plt.title("Training Cost", fontsize=24)
plt.xlabel("Iter", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.plot(iters, costs, color="red", label="Training Cost")
plt.grid()

# 线性模型可视化
plt.figure("Inference")
plt.title("Linear Regression", fontsize=24)
plt.plot(x_test, y_out[0], color="red", label="inference")
plt.scatter(train_data, y_true)
plt.legend()
plt.grid()
plt.show()
