import paddle.fluid as fluid

x = fluid.layers.fill_constant(shape=[1],dtype="int64",value=5)
y = fluid.layers.fill_constant(shape=[1],dtype="int64",value=1)
z = x+y

# z执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 初始化
exe.run(program=fluid.default_startup_program())
# 执行
out = exe.run(program=fluid.default_main_program(),fetch_list=[z])
print(out)