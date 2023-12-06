#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  add_demo_01.py
:time  2023/12/5 14:55
:desc  
"""
import paddle.fluid as fluid

# 定义两个张量
x = fluid.layers.fill_constant(shape=[1],
                               dtype="int64",
                               value=5)
y = fluid.layers.fill_constant(shape=[1],
                               dtype="int64",
                               value=1)
z = x + y
# 执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)

result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[z])
print(type(result[0]))
print(result[0])
