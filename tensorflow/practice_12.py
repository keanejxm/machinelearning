#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_12.py
:time  2023/2/10 16:59
:desc  AutoGraph和tf.Module
"""
import tensorflow as tf
import numpy as np

# 定义一个简单的function

x = tf.Variable(1.0, dtype=tf.float32)


# 在tf.function中用input_signature限定输入张量的签名类型：shape和dtype
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return (x)


add_print(tf.constant(3.0, dtype=tf.float32))


# 利用tf.Module的子类将其封装一下
class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name)
        with self.name_scope:  # 定义一个命名空间
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def add_print(self, a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return (self.x)


if __name__ == '__main__':
    # 创建日志
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "./data/demomodule/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)
    # 开启AutoGraph跟踪
    tf.summary.trace_on(graph=True, profiler=True)

    obj = DemoModule(init_value=tf.constant(1.0))
    result = obj.add_print(tf.constant(3.0))

    # 将计算图写入日志
    with writer.as_default():
        tf.summary.trace_export(
            name="demomodule",
            step=0,
            profiler_outdir=logdir
        )
    # 查看模块中的全部变量和全部可训练变量
    print(obj.variables)
    print(obj.trainable_variables)
    # 查看模块中的子模块
    print(obj.submodules)
    # 使用tf.save_module.save保存模型，并指定需要跨平台部署的方法
    tf.saved_model.save(obj, "./data/deom/1", signatures={"serving_default": obj.add_print})

    # 加载模型
    demo2 = tf.saved_model.load("./data/deom/1")
    demo2.add_print(tf.constant(2.0))
