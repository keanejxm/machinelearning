#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  05_paddle_boston_demo.py
:time  2023/12/6 16:03
:desc  波士顿房价预测
"""
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import matplotlib.pyplot as plt

# 数据准备
BUF_SIZE = 500  # 随机
BATCH_SIZE = 20  # 批次
reader = paddle.dataset.uci_housing.train()
random_reader = paddle.reader.shuffle(reader, buf_size=BUF_SIZE)
batch_reader = paddle.batch(random_reader, batch_size=BATCH_SIZE)
# for data in batch_reader():
#     print(data)
#     break
# 构建模型
x = fluid.layers.data(name="x",
                      shape=[13],
                      dtype="float32")
y = fluid.layers.data(name="y",
                      shape=[1],
                      dtype="float32")
y_predict = fluid.layers.fc(input=x,
                            size=1,
                            act=None)
# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict,
                                      label=y)
avg_cost = fluid.layers.mean(cost)  # 均方差损失函数

# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 训练保存模型
# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())

# feeder
feeder = fluid.DataFeeder(place=place,
                          feed_list=[x, y])
#
iter = 0
iters = []
train_costs = []
for epoch in range(120):
    c = 0  # 接受返回的损失值
    i = 0
    for data in batch_reader():
        i += 1
        c = exe.run(program=fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost])
        if i % 10 == 0:
            print(f"epoch:{epoch},batch:{i},cost:{c[0][0]}")
        iter += BATCH_SIZE
        iters.append(iter)
        train_costs.append(c[0][0])
# 保存模型
model_save_dir = "./model/inference_model/uci_housing"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
# 保存推理模型
# 推理模型：只包含器前向传播部分，不包含反向传播部分
fluid.io.save_inference_model(model_save_dir,
                              ["x"],
                              [y_predict],
                              exe)
# 可视化
plt.figure("Training Cost")
plt.title("Training Cost", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.plot(iters, train_costs, color="red", label="Training Cost")
plt.grid()
plt.legend()
plt.savefig("cost.png")
# plt.show()
# 加载模型测试
# 定义预测执行器
infer_exe = fluid.Executor(place)
# infer_prog:专门用于推理的program
# feed_vars:推理时需要输入的张量名称
# fetch_target:预测结果从哪里获取
infer_prog, feed_vars, fetch_targes = fluid.io.load_inference_model(
    dirname=model_save_dir,
    executor=infer_exe
)
infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),batch_size=200)
test_data = next(infer_reader())
test_x = np.array([d[0] for d in test_data]).astype("float32")
test_y = np.array([d[1] for d in test_data]).astype("float32")
# 参数字典
params = {feed_vars[0]:test_x}
# 执行推理
results = infer_exe.run(program=infer_prog,
                        feed=params,
                        fetch_list=fetch_targes)
print(results)
# 模型评估
# 预测
