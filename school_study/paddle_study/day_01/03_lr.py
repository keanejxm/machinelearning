# paddle实现线性回归
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]]).astype('float32')
y_true = np.array([[5.0], [5.5], [6.0], [6.8], [6.8]]).astype("float32")
x = fluid.layers.data(name="x", shape=[1], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")

y_predict = fluid.layers.fc(input=x,
                            size=1,
                            act=None)
# 损失函数
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)  # 均方差
# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(avg_cost)

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())  # 初始化

costs = []  # 损失值
iters = []  # 存放迭代次数
values = []
params = {"x": train_data, "y": y_true}
for i in range(200):  # 循环训练
    outs = exe.run(program=fluid.default_main_program(),  # 执行的program
                   feed=params,  # 参数字典
                   fetch_list=[y_predict, avg_cost])  # 返回与测试、损失值
    print(outs)
    costs.append(outs[1][0])
    iters.append(i)  # 存放迭代次数
    print("i:", i, " cost:", outs[1][0])
# 损失函数可视化
plt.figure("Training")
plt.title("TrainingCost", fontsize=14)
plt.xlabel("Iter", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.plot(iters, costs, color="red", label="Training Cost")  # 绘制损失函数曲线
plt.grid()  # 绘制网格线
# plt.show()
plt.savefig("train.png")
# 线性模型可视化
tmp = np.random.rand(10, 1)
tmp = tmp * 2
tmp.sort(axis=0)
x_test = np.array(tmp).astype("float32")
params = {"x": x_test, "y": x_test}
y_out = exe.run(feed=params, fetch_list=[y_predict.name])
y_test = y_out[0]

# 线性模型可视化
plt.figure("Inference")
plt.title("Line Regression", fontsize=24)
plt.plot(x_test, y_test, color="red", label="inference")
plt.scatter(train_data, y_true)

plt.legend()
plt.grid()
plt.savefig("infer.png")
plt.show()
