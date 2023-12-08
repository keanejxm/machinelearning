#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  05_paddle_fruits_cnn_demo.py
:time  2023/12/7 14:36
:desc  水果分类
"""
import os

from common_utils import *

# 数据处理，划分测试集训练集
data_root_path = f"{DATA_PATH}/fruits"

name_dict = {
    "apple": 0,
    "banana": 1,
    "grape": 2,
    "orange": 3,
    "pear": 4
}

test_file = "./fruits/test.txt"
train_file = "./fruits/train.txt"
name_data_list = {}


def save_to_dict(path, name):
    """
    将图片路径存入字典
    :param path: 图片路径
    :param name: 图片类别
    :return:
    """
    if name not in name_data_list:
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list
    else:
        name_data_list[name].append(path)


# 遍历图片添加到测试集训练集
sub_dir = os.listdir(data_root_path)
for d in sub_dir:
    full_path = os.path.join(data_root_path, d)
    if os.path.isdir(full_path):
        imgs = os.listdir(full_path)
        for fn in imgs:
            img_full_path = os.path.join(full_path, fn)
            save_to_dict(img_full_path, d)
    else:
        continue

# 清空训练集测试集文件
with open(test_file, "w") as w1:
    pass
with open(train_file, "w") as w2:
    pass

for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)
    print(f"{name}:{num}张")
    for img in img_list:
        line = f"{img}\t{name_dict[name]}\n"
        if i % 10 == 0:
            # 写入测试集
            with open(test_file, "a") as at:
                at.write(line)
        else:
            # 写入训练集
            with open(train_file, "a") as ar:
                ar.write(line)
        i += 1
print(f"划分测试集，训练集结束")

# 构建模型
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    img_path, label = sample
    img = paddle.dataset.image.load_image(img_path)  # 读取图像数据
    img = paddle.dataset.image.simple_transform(
        im=img,  # 原图像
        resize_size=128,  # 缩放至128*128
        crop_size=128,  # 裁剪大小
        is_color=True,  # 彩色图像
        is_train=True  # 训练模式，训练模式下会随机裁剪
    )
    # 图像归一化
    img = img.astype("float32") / 255
    return img, label


def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]
            for ln in lines:
                ln = ln.replace("\n", "")
                img_path, lbl = ln.split("\t")
                yield img_path, lbl

    return paddle.reader.xmap_readers(
        train_mapper,  # 下一步处理函数
        reader,
        cpu_count(),  # 线程数量（和cpu数量一致）
        buffered_size  # 缓冲区大小
    )


# 构建模型
def create_cnn(image, type_size):
    """
    创建cnn
    :param image:
    :param type_size:
    :return:
    """
    # 第一层卷积池化
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,
        filter_size=3,  # 3*3卷积核
        num_filters=32,  # 卷积核数量
        pool_size=2,  # 2*2池化
        pool_stride=2,  # 池化步长
        act="relu"
    )
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二层卷积池化
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,
        filter_size=3,  # 3*3卷积核
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 2*2池化
        pool_stride=2,  # 池化步长
        act="relu"
    )
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)
    # 第二层卷积池化
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,
        filter_size=3,  # 3*3卷积核
        num_filters=64,  # 卷积核数量
        pool_size=2,  # 2*2池化
        pool_stride=2,  # 池化步长
        act="relu"
    )
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # fc层
    fc = fluid.layers.fc(input=drop, size=512, act="relu")
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输入层
    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict


# 定义reader
BATCH_SIZE = 32
train_reader = train_r(train_list=train_file)
random_train_reader = paddle.reader.shuffle(reader=train_reader,
                                            buf_size=1300)
batch_train_reader = paddle.batch(
    random_train_reader,
    batch_size=BATCH_SIZE
)
# 张量占位符
image = fluid.layers.data(name="image",
                          shape=[3, 128, 128],
                          dtype="float32")
label = fluid.layers.data(name="label",
                          shape=[1],
                          dtype="int64")
predict = create_cnn(image, type_size=5)
# 损失函数
cost = fluid.layers.cross_entropy(input=predict,
                                  label=label)
avg_cost = fluid.layers.mean(cost)
# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 准确率
accuracy = fluid.layers.accuracy(input=predict,
                                 label=label)

# 执行器
place = fluid.CPUPlace()
# place = fluid.CUDAPlace()
exe = fluid.Executor(place)
# 初始化
exe.run(fluid.default_startup_program())
# 数据喂入器
feeder = fluid.DataFeeder(feed_list=[image, label],
                          place=place)
# 训练模型
epochs = []
costs = []
accs = []
times = 0
batches = []
for epoch in range(10):
    for batch_id, data in enumerate(batch_train_reader()):
        times += 1
        c, a = exe.run(program=fluid.default_main_program(),
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, accuracy])
        if batch_id % 10 == 0:
            print(f"epoch:{epoch},batch:{batch_id},cost:{c[0]},acc:{a[0]}")
            accs.append(a[0])
            costs.append(c[0])
            batches.append(times)

# 保存模型
model_save_dir = "../model/fruits/"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=["image"],
                              target_vars=[predict],
                              executor=exe)
print("模型保存完成")
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color="red", label="Training Cost")
plt.plot(batches, accs, color="green", label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()
# 测试
from PIL import Image
# 读取测试图像
def load_img(path):
    img = paddle.dataset.image.load_and_transform(path,
                                                  128,128,
                                                  False)
    img = img.astype("float32")/255.0
    return img

# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = "../model/fruits/"
# 加载模型
infer_prog ,feed_vars,fetch_targets = fluid.io.load_inference_model(
    model_save_dir,infer_exe
)
test_img = ""
infer_imgs = []
infer_imgs.append(load_img(test_img))
infer_imgs = numpy.array(infer_imgs)
# 执行预测
params = {feed_vars[0]:infer_imgs}
result = infer_exe.run(infer_prog,feed=params,fetch_list=fetch_targets)
print(result)
# 评估
