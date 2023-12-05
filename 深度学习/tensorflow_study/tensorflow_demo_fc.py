#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_cnn.py
:time  2023/12/4 11:16
:desc  手写数字识别，全连接神经网络
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
from common_utils import *

# 读取数据集

mnist = input_data.read_data_sets(f"{DATA_PATH}/MNIST_data", one_hot=True)
print(mnist)

# 占位符
x = tf.placeholder("float32", shape=[None, 784])
y = tf.placeholder("float32", shape=[None, 10])

# 定义权重、偏置
init_w = tf.random_normal(shape=[784, 10])
w = tf.Variable(init_w)
init_b = tf.zeros(shape=(10,))
b = tf.Variable(init_b)

# 搭建模型
pred_y = tf.nn.softmax(tf.matmul(x, w) + b)

# 构建损失函数(交叉熵：真实值减去预测值的对数然后求平均值)
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y), reduction_indices=1)
# 求损失函数最小值
cost = tf.reduce_mean(cross_entropy)
# 梯度下降
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

batch_size = 100
saver = tf.train.Saver()

# 执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 在模型训练之前判断是否存在已训练模型，如果有加载
    if os.path.exists("../model/mnist/checkpoint"):
        saver.restore(sess, "../model/mnist/")

    # 训练:外层控制轮数，内存控制批次
    for epoch in range(10):
        total_batch = int(mnist.train.num_examples / batch_size)
        total_cost = 0.0
        for i in range(total_batch):
            train_x, train_y = mnist.train.next_batch(batch_size)
            params = {x: train_x, y: train_y}
            o, c = sess.run([train_op, cost], feed_dict=params)
            total_cost += c

        avg_cost = total_cost / total_batch
        print(f"轮数：{epoch},cost：{avg_cost}")
    print("训练结束")

    # 模型评估
    corr = tf.equal(tf.argmax(y, 1),
                    tf.argmax(pred_y, 1))

    accuracy = tf.reduce_mean(tf.cast(corr, "float32"))

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(f"精度：{acc}")

    # 保存模型
    saver.save(sess, "../model/mnist/")

# 加载模型预测

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 加载模型
    saver.restore(sess, "../model/mnist/")
    # 从测试中随机取到2张图像
    test_x, test_y = mnist.test.next_batch(2)

    # 执行预测
    predv = sess.run(pred_y, feed_dict={x: test_x})

    out_put = tf.argmax(predv, 1).eval()
    y_true = tf.argmax(test_y, 1).eval()
    pred_val = tf.reduce_max(predv, 1).eval()
    print(f"真实类别:{y_true}")
    print(f"预测类别:{out_put}")
    print(f"预测概率:{pred_val}")

    # 显示图像
    import pylab

    img1 = test_x[0].reshape(28, 28)
    pylab.imshow(img1)
    pylab.show()

    img2 = test_x[1].reshape(28, 28)
    pylab.imshow(img2)
    pylab.show()
