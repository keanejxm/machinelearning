#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  news_class.py
:time  2024/3/15 15:11
:desc  新闻分类
"""

import os
from multiprocessing import cpu_count
import numpy as np
import paddle
import paddle.fluid as fluid

# 定义一组公共变量
data_root = "data/"  # 数据集所在目录
data_file = "news_classify_data.txt"  # 原始数据集
train_file = "train.txt"  # 训练集文件
test_file = "test.txt"  # 测试集文件
dict_file = "dict_txt.txt"  # 字典文件

data_file_path = os.path.join(data_root, data_file)  # 数据集完成路径
train_file_path = os.path.join(data_root, train_file)  # 训练集文件完整路径
test_file_path = os.path.join(data_root, test_file)  # 测试集文件完整路径
dict_file_path = os.path.join(data_root, dict_file)  # 字典文件完成路径


# 取出样本中所有的字，对每个字进行编码，将编码结果存入字典文件
def create_dict():
    dict_set = set()  # 集合，用作去重
    with open(data_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace("\n", "")  # 去除换行符
            tmp_list = line.split("_!_")  # 根据分割符拆分
            title = tmp_list[-1]  # 最后一个字段为标题
            for word in line:
                dict_set.add(word)
    # 遍历集合，取出每个字进行编号
    dict_txt = {}
    i = 1
    for word in dict_set:
        dict_txt[word] = i
        i += 1
    dict_txt["<unk>"] = i  # 未知字符，未在样本中出现过的字

    # 将字典文件存储
    with open(dict_file_path, "w", encoding="utf-8") as f:
        f.write(str(dict_txt))
    print("生成字典文件结束")


# 传入一个句子，将每个字换为编码值，与标签一起返回
def line_encoding(title, dict_txt, label):
    new_line = ""  # 编码结果
    for word in title:
        if word in dict_txt:
            new_line += str(dict_txt[word])
        else:
            new_line += str(dict_txt["<unk>"])
        new_line += ","
    new_line = new_line[:-1]
    new_line = new_line + "\t" + label + "\n"
    return new_line


def create_train_test_file():
    # with open(train_file_path, "w", encoding="utf-8") as f:
    #     pass
    # with open(test_file_path, "w", encoding="utf-8") as f:
    #     pass

    # 读取字典文件
    with open(dict_file_path, "r", encoding="utf-8") as f_dict:
        dict_txt = eval(f_dict.readlines()[0])

    # 读取原始样本
    with open(data_file_path, "r", encoding="utf-8") as f_data:
        lines = f_data.readlines()

    i = 0
    for line in lines:
        tmp_list = line.replace("\n", "").split("_!_")
        title = tmp_list[-1]
        label = tmp_list[1]
        new_line = line_encoding(title, dict_txt, label)
        if i % 10 == 0:  # 写入测试集
            with open(test_file_path, "a", encoding="utf-8") as f:
                f.write(new_line)
        else:
            with open(train_file_path, "a", encoding="utf-8") as f:
                f.write(new_line)
        i += 1

    print("生成训练集测试集结束")


create_dict()
create_train_test_file()


def get_dict_len(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0])
    return len(dict_txt.keys())


def data_mapper(sample):
    data, label = sample
    val = [int(w) for w in data.split(",")]
    return val, int(label)


def train_reader(train_file_path):
    def reader():
        with open(train_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            np.random.shuffle(lines)

            for line in lines:
                data, label = line.split("\t")
                yield data, label

    return paddle.reader.xmap_readers(
        mapper=data_mapper,
        reader=reader,
        process_num=cpu_count(),
        buffer_size=1024
    )


def test_reader(test_file_path):
    def reader():
        with open(test_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split("\t")
                yield data, label

    return paddle.reader.xmap_readers(
        mapper=data_mapper,
        reader=reader,
        process_num=cpu_count(),
        buffer_size=1024
    )


# 定义网络
def lstm_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=128):
    """

    :param data:
    :param dict_dim:
    :param class_dim:
    :param emb_dim: 词嵌入长度
    :param hid_dim: 第一个卷积层卷积核数量
    :param hid_dim2: 第二个卷积层卷积核数量
    :return:
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim]
    )
    conv1 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act='tanh',
        pool_type="sqrt"
    )
    conv2 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim2,
        filter_size=4,
        act='tanh',
        pool_type='sqrt'
    )
    output = fluid.layers.fc(
        input=[conv1, conv2],
        size=class_dim,
        act="softmax"
    )
    return output


words = fluid.layers.data(
    name="words",
    shape=[1],
    dtype='int64',
    lod_level=1
)
label = fluid.layers.data(
    name="label",
    shape=[1],
    dtype="int64"
)
dict_dim = get_dict_len(dict_file_path)

model = lstm_net(words, dict_dim)

# 损失函数
cost = fluid.layers.cross_entropy(input=model, label=label)

avg_cost = fluid.layers.mean(cost)

# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
optimizer.minimize(avg_cost)

# 准确率
accuracy = fluid.layers.accuracy(input=model, label=label)

# 执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

tr_reader = train_reader(train_file_path)
batch_train_reader = paddle.batch(reader=tr_reader, batch_size=128)
ts_reader = test_reader(test_file_path)
batch_test_reader = paddle.batch(reader=ts_reader, batch_size=128)

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

for epoch in range(80):
    for batch_id, data in enumerate(batch_train_reader):
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost, accuracy]
        )
        if batch_id % 100 == 0:
            print("epoch:%d,batch:%d,cost:%f,acc:%f" % (epoch, batch_id, train_cost[0], train_acc[0]))
    # 评估
    test_costs_list = []  # 测试集损失值
    test_accs_list = []  # 测试集准确率
    for batch_id, data in enumerate(batch_test_reader):
        test_cost, test_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost, accuracy]
        )
        test_costs_list.append(test_cost[0])
        test_accs_list.append(test_acc[0])
    avg_test_cost = sum(test_costs_list) / len(test_costs_list)
    avg_test_acc = sum(test_accs_list) / len(test_accs_list)
    print("epoch:%d,test_cost:%f,test_acc:%f" % (epoch, avg_test_cost, avg_test_cost))

# 保存模型
model_save_dir = "model/"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir, feeded_var_names=[words.name], target_vars=[model], executor=exe)
print("模型保存成功")


# 推理预测

def get_data(sentence):
    with open(dict_file_path, "r", encoding="utf-8") as f:
        dict_txt = eval(f.readlines()[0])
        ret = []
        for w in sentence:
            if w not in dict_txt:
                ret.append(dict_txt["<unk>"])
            else:
                ret.append(input(dict_txt[w]))
    return ret


place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
infer_program, feed_names, target_var = fluid.io.load_inference_model(dirname=model_save_dir, executor=exe)

tests = []

data = get_data("")

base_shape = [[len(c) for c in tests]]
tensor_words = fluid.create_lod_tensor(
    data=tests,
    recursive_seq_lens=base_shape,
    place=place
)
result = exe.run(
    program=infer_program,
    feed={feed_names[0]: tensor_words},
    fetch_list=target_var
)
names = ["文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "国际", "证券"]
for r in result[0]:
    idex = np.argmax(r)  # 去除最大值索引
    print("预测结果：", names[idex], "概率：", r[idex])
