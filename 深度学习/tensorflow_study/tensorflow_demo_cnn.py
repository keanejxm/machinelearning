#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  tensorflow_demo_cnn.py
:time  2023/12/4 15:05
:desc  使用卷积实现服饰识别
"""
import os.path

import tensorflow as tf
from common_utils import *
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


class FashionMnist:
    out_feature1 = 12  # 第一组的卷积核数量
    out_feature2 = 24  # 第二组卷积核数量
    con_neurons = 512  # 全连接层神经元数量

    def __init__(self, path):
        self.data = read_data_sets(path, one_hot=True)
        self.sess = tf.Session()

    def close(self):
        self.sess.close()

    # 初始化权重
    def init_weight_var(self, shape):
        # 截尾正态分布
        init_w = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_w)

    # 初始化偏置
    def init_bias_var(self, shape):
        init_b = tf.constant(1.0, shape=shape)
        return tf.Variable(init_b)

    # 二维卷积
    def conv2d(self, x, w):
        return tf.nn.conv2d(
            x,
            w,  # 卷积核
            strides=[1, 1, 1, 1],  # 步长
            padding='SAME'  # 填充
        )

    # 池化
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(
            x,  # 输入数据
            ksize=[1, 2, 2, 1],  # 池化区域
            strides=[1, 2, 2, 1],  # 池化步长
            padding="SAME"
        )

    # 卷积池化组
    def create_conv_layer(self, input, input_c, out_c):
        """
        卷积池化组 5*5 卷积
        :param input: 输入数据
        :param input_c: 输入通道数
        :param out_c: 输出通道数
        :return:
        """
        # 卷积核
        filter_w = self.init_weight_var([5, 5, input_c, out_c])
        # 卷积核的偏置
        b_conv = self.init_bias_var([out_c])
        # 执行卷积激活
        h_conv = tf.nn.relu(self.conv2d(input, filter_w) + b_conv)
        # 执行池化
        h_pool = self.max_pool_2x2(h_conv)
        return h_pool

    # 全连接层
    def create_fc_layer(self, h_pool_flat, input_feature, con_neurons):
        """
        全连接层
        :param h_pool_flat: 输入数据（1维）
        :param input_feature: 输入的特征数量
        :param con_neurons: 神经元数量
        :return:
        """
        w_fc = self.init_weight_var([input_feature, con_neurons])
        b_fc = self.init_bias_var([con_neurons])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
        return h_fc1

    # 构建cnn
    def build(self):
        """"""
        # 样本数据占位符
        self.x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # 第一组卷积池化
        h_pool1 = self.create_conv_layer(x_image,
                                         1,
                                         self.out_feature1)
        # 第二组卷积池化
        h_pool2 = self.create_conv_layer(h_pool1,
                                         self.out_feature1,
                                         self.out_feature2)
        # 全连接层
        h_pool2_flat_feature = 7 * 7 * self.out_feature2
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_feature])

        h_fc = self.create_fc_layer(h_pool2_flat,
                                    h_pool2_flat_feature,
                                    self.con_neurons)

        # dropout层
        h_fc_drop = tf.nn.dropout(h_fc, 0.5)

        # 输出层
        w_fc = self.init_weight_var([self.con_neurons, 10])
        b_fc = self.init_bias_var([10])

        self.pred_y = tf.matmul(h_fc_drop, w_fc) + b_fc

        # 损失函数
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                       logits=self.pred_y)
        # 求均值
        cross_entropy = tf.reduce_mean(loss)

        # 梯度下降：自适应梯度下降优化器
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        # 精度
        corr = tf.equal(tf.argmax(self.y, 1),
                        tf.argmax(self.pred_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))

    # 训练
    def train(self):
        # 初始化
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 训练之前检查是否存在模型，存在模型加载模型
        if os.path.exists("../model/fashion_mnist/checkpoint"):
            saver.restore(self.sess, "../model/fashion_mnist/")
        # 批次大小
        batch_size = 100
        print("开始训练")
        for i in range(10):  # 轮数
            total_batch = int(self.data.train.num_examples / batch_size)
            total_acc = 0.0
            for j in range(total_batch):
                train_x, train_y = self.data.train.next_batch(batch_size)

                params = {self.x: train_x, self.y: train_y}
                t, acc = self.sess.run([self.train_op, self.accuracy],
                                       feed_dict=params)
                # 平均精度
                total_acc += acc
            avg_acc = total_acc / total_batch
            print(f"轮数：{i + 1},精度：{avg_acc}")
        # 保存模型
        saver.save(self.sess, "../model/fashion_mnist/")

    def metrics(self):
        # 拿到测试集的数据
        test_x, test_y = self.data.test.next_batch(10000)
        params = {self.x: test_x, self.y: test_y}
        test_acc = self.sess.run(self.accuracy, feed_dict=params)
        print(f"测试集精度：{test_acc}")

    def pred_fashion(self):
        test_x, test_y = self.data.test.next_batch(1)
        params = {self.x: test_x}
        pred_y = self.sess.run(self.pred_y, feed_dict=params)

        out_put = tf.argmax(pred_y, 1).eval()
        y_true = tf.argmax(test_y, 1).eval()
        pred_val = tf.reduce_max(pred_y, 1)

        print(f"预测：{out_put}")
        print(f"真实：{y_true}")
        print(f"概率：{pred_val}")


if __name__ == '__main__':
    mnist = FashionMnist(f"{DATA_PATH}/fashion_mnist")
    mnist.build()
    mnist.train()
    mnist.metrics()
    mnist.close()
    pred_mnist = FashionMnist(f"{DATA_PATH}/fashion_mnist")
    pred_mnist.pred_fashion()
