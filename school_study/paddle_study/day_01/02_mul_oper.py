import numpy as np
import paddle.fluid as fluid
import numpy

# 定义两个张量
x = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")

# 计算两个张量的和
x_add_y = fluid.layers.elementwise_add(x, y)
# 计算两个张量的乘积
x_mul_y = fluid.layers.elementwise_mul(x, y)

# （数据） 给两个张量定义2，3的数组

# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())  # 初始化

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 1, 1], [2, 2, 2]])

params = {"x": a, "y": b}
# 执行
outs = exe.run(program=fluid.default_main_program(),
        feed=params,
        fetch_list=[x_add_y, x_mul_y])
print(outs[0])