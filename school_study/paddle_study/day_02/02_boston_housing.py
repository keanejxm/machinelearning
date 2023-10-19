# 利用神经网络（fc）实现波士顿房价预测
# 输入：13个特征值
# 输出（标签）：房屋价格
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import matplotlib.pyplot as plt

# 准备数据
# 定义模型
# 损失函数
# 训练
# 保存模型
# 加载模型，用测试集进行测试

# 缓冲区大小
BUF_SIZE = 500
# 批次大小
BATCH_SIZE = 20
# 定义训练集
reader = paddle.dataset.uci_housing.train()
shuffle_reader = paddle.reader.shuffle(reader, BATCH_SIZE)
batch_reader = paddle.batch(shuffle_reader, BATCH_SIZE)

# for data in batch_reader():
#     print(data[0][0])
x = fluid.layers.data(
    name="x",
    shape=[13],
    dtype="float32")
y = fluid.layers.data(
    name="y",
    shape=[1],
    dtype="float32")
y_predict = fluid.layers.fc(
    input=x,
    size=1,
    act=None
)
# 预测值与真实值构建损失函数

cost = fluid.layers.square_error_cost(input=y_predict,
                                      label=y)

# 均方差
avg_cost = fluid.layers.mean(cost)

# 优化器
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
# 指定优化的目标函数
optimizer.minimize(avg_cost)

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())

# feeder
feeder = fluid.DataFeeder(
    place=place,
    feed_list=[x, y])
# 迭代的批次
iter = 0
iters = []
train_costs = []
for epoch in range(120):
    c = 0
    i = 0
    for data in batch_reader():
        i += 1
        c = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost]
        )
        if i % 10 == 0:
            print(f"epoch:{epoch},batch:{i},cost:{c[0][0]}")
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(c[0][0])
# 保存模型
model_save_dir = "./uci_housing"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

# 保存推理模型（最终用于预测的模型）
# 推理模型：只包含前向传播部分，不需要反向传播部分
fluid.io.save_inference_model(
    model_save_dir,
    ["x"],  # 推理时传入的张量的名称
    [y_predict],  # 推理结果从哪里获取
    exe
)

# 可视化
plt.figure("Training Cost")
plt.title("Train Cost", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.plot(iters, train_costs, color="red", label="Training Cost")
plt.grid()
plt.savefig("uci_housing.png")
plt.show()

# 加载模型
infer_exe = fluid.Executor(place)  # 专门用于推理的执行器

# infer_prog:专门用于推理的program
# feed_vars:推理
# fetch_targets:预测结果从哪里获取
infer_prog, feed_vars, fetch_targets = fluid.io.load_inference_model(
    model_save_dir,
    infer_exe
)

infer_reader = paddle.batch(
    paddle.dataset.uci_housing.test(),
    batch_size=200
)
test_data = next(infer_reader())

test_x = np.array([d[0] for d in test_data]).astype("float32")
test_y = np.array([d[1] for d in test_data]).astype("float32")

params = {feed_vars[0]: test_x}
# 执行推理
results = infer_exe.run(
    program=infer_prog,
    feed=params,
    fetch_list=fetch_targets
)
print(results[0].shape)

infer_result = []
ground_truth = []

for val in results[0]:
    infer_result.append(val)

for val in test_y:
    ground_truth.append(val)

plt.figure("scatter")
plt.title("TestFigure", fontsize=24)
plt.xlabel("ground truth", fontsize=14)
plt.ylabel("infer truth", fontsize=14)
x = np.arange(1, 30)
y = x
plt.plot(x, y)
plt.scatter(ground_truth, infer_result, color="green", label="Test")
plt.grid()
plt.legend()  # 绘制图例
plt.savefig("predict.png")
plt.show()
