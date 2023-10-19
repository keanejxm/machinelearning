"""
线性回归
"""
import tensorflow as tf
import os

# 数据准备
x = tf.random_normal(shape=(100, 1), mean=1.75, stddev=0.5)

# 输出数据（真实）y = 2x+5
y = tf.matmul(x, [[2.0]]) + 5.0

# 搭建模型,weight(权重) ，偏置bias
init_w = tf.random_normal([1, 1])
weight = tf.Variable(init_w, trainable=True)  # trainable:是否是训练模式，参数是否可以发生变化
bias = tf.Variable(0.0, trainable=True)

predict_y = tf.matmul(x, weight) + bias

# 有了预测值，创建损失函数（回归，选择均方误差）
loss = tf.reduce_mean(tf.square(y - predict_y))
# 梯度下降，求损失函数最小值（使用梯度下降的优化器）
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)  # 求损失函数的极小值

# 收集损失函数的值
# 收集标量
tf.summary.scalar("losses", loss)
merged = tf.summary.merge_all()  # 合并摘要

# 保存模型
saver = tf.train.Saver()  # var_list:保存的模型参数变量，None保存全部变量，max_to_keep = 5:保存的最大的检查点文件

# 执行
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    #
    fw = tf.summary.FileWriter("../summary/", graph=sess.graph)
    # 开始训练之前判断是否有模型保存，有，加载模型
    if os.path.exists("../model/lr/checkpoint"):
        saver.restore(sess, "../model/lr/")

    for i in range(100):
        sess.run(train_op)  # 执行一次梯度下降，收集一次损失值
        summary = sess.run(merged)  # 收集损失值
        fw.add_summary(summary, i)
        print("轮数：{}，w:{}，b:{}".format(i, weight.eval(), bias.eval()))

    saver.save(sess, "../model/lr/")  # 保存模型
