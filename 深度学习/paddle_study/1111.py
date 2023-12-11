#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  1111.py
:time  2023/12/11 20:26
:desc  
"""
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt

# 根据传入的文本样本(样本由图片路径、所属类别两个值构成)
# 读取图片数据，并进行归一化处理，再返回
def train_mapper(sample):
    img, label = sample   # 将sample中的值赋值给img, label
    if not os.path.exists(img):
        print(img, "文件不存在")
    # 读取图片
    img_data = paddle.dataset.image.load_image(img)
    # 对读取的图片数据进行变换(简单修剪)， 输出固定大小的图像数据
    img_data = paddle.dataset.image.simple_transform(im=img_data, # 图像数据
                                                     resize_size=100, # 100*100
                                                     crop_size=100,# 100*100
                                                     is_color=True, #彩色图像
                                                     is_train=True)
    # 对图像进行平滑处理, 归一化处理(0~1)
    img_data = img_data.flatten().astype("float32") / 255.0

    return img_data,
    # 返回图像、标签

# 定义reader
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f: # 打开训练集文件
            lines = [line.strip() for line in f]  # 读取所有文件行
            for line in lines:
                line = line.replace("\n", "") # 去掉换行符
                img_path, lab = line.split("\t")  # 根据tab符号进行拆分
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper, # 读取图片数据函数
                                      reader, # 原始读取样本reader
                                      cpu_count(), # 线程数量
                                      buffered_size) # 缓冲区大小

# 准备数据
BATCH_SIZE = 32  # 批次大小
train_reader = train_r(train_list=train_file_path) # 构建读取器
shuffle_reader = paddle.reader.shuffle(reader=train_reader,
                                       buf_size=1300)
batch_train_reader = paddle.batch(shuffle_reader,
                                  batch_size=BATCH_SIZE)#批量读取器

# 定义变量
image = fluid.layers.data(name="image", shape=[3, 100, 100], dtype="float32")
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# 搭建VGG模型
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_fiter, groups, dropouts):
        return fluid.nets.img_conv_group(input=ipt, # 输入图像, 格式[N,C,H,W]
                                         pool_size=2, # 池化区域大小
                                         pool_stride=2, # 池化步长
                                         conv_num_filter=[num_fiter] * groups,
                                         conv_filter_size=3, # 过滤器大小
                                         conv_act="relu", # 激活函数
                                         conv_with_batchnorm=True,
                                         conv_batchnorm_drop_rate=dropouts,
                                         pool_type="max") # 池化类型
    # 五个卷积池化组
    conv1 = conv_block(image, 64, 2, [0.0, 0]) # 第一个卷积池化组
    conv2 = conv_block(conv1, 128, 2, [0.0, 0]) # 第二个卷积池化组
    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0]) # 第三个卷积池化组
    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0]) # 第四个卷积池化组
    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0]) # 第五个卷积池化组
    # drop
    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.2)
    # fc
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    # batch normal
    bn = fluid.layers.batch_norm(input=fc1, act="relu")
    # drop
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)
    # fc
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    # output
    predict = fluid.layers.fc(input=fc2, size=type_size, act="softmax")
    return predict

# 调用函数，创建CNN模型
predict = vgg_bn_drop(image=image, type_size=5)
# 定义损失函数
cost = fluid.layers.cross_entropy(input=predict, # 预测值(一组概率)
                                  label=label) # 真实值(一组概率)
avg_cost = fluid.layers.mean(cost) # 求均值
# 计算准确率
accuracy = fluid.layers.accuracy(input=predict, # 预测值
                                 label=label) # 真实值
# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.000001) # 自适应梯度下降优化器
# optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer.minimize(avg_cost)

# 执行器
# place = fluid.CPUPlace() # 如果不能使用GPU，则写这一句
place = fluid.CUDAPlace(0) # 如果可以使用GPU，则写这一句
exe = fluid.Executor(place) # 执行器
exe.run(fluid.default_startup_program()) # 初始化
# 定义feeder
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

# 训练
costs = [] # 记录损失值
accs = [] # 记录准确率
times = 0 # 次数
batches = []  # 存放训练轮次

# 加载增量模型
persis_model_dir = "persis_model/"
if os.path.exists(persis_model_dir):
    fluid.io.load_persistables(exe, persis_model_dir, fluid.default_main_program())
    print("加载增量模型成功.")

print("开始训练......")
for pass_id in range(100):
    train_cost = 0
    for batch_id, data in enumerate(batch_train_reader()): # 循环取一个批次的数据样本
        times += 1
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data), # 喂入参数
                                        fetch_list=[avg_cost, accuracy])#获取结果
        if batch_id % 20 == 0:
            print("passid:%d, batchid:%d, cost:%f, acc:%f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
            accs.append(train_acc[0])
            costs.append(train_cost[0])
            batches.append(times)
# 保存增量模型
if not os.path.exists(persis_model_dir):  # 如果存储模型的目录不存在，则创建
    os.makedirs(persis_model_dir)
fluid.io.save_persistables(exe, persis_model_dir, fluid.default_main_program())

# 保存模型
model_save_dir = "model/fruits/"

if not os.path.exists(model_save_dir): # 模型保存路径不存在
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=["image"],
                              target_vars=[predict],
                              executor=exe)
print("模型保存完成.")

# 训练过程可视化
plt.figure('training', facecolor='lightgray')
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label="Training Cost")
plt.plot(batches, accs, color='green', label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("Train.png")
plt.show()