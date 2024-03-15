#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  emotion_analyse.py
:time  2024/3/14 15:10
:desc  
"""
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np
import os
import random
from multiprocessing import cpu_count
from common_utils import DATA_PATH

mydict = {}
code = 1
# data_file = "data/hotel_discuss2.csv"
# data_file = "data/hotel_discuss2.csv"
# dict_file = "data/hotel_dict.txt"
# encoding_file = "data/hotel_encoding.txt"
data_path = os.path.join(DATA_PATH, "nlp_data")
data_file = os.path.join(data_path, "hotel_discuss2.csv")
dict_file = os.path.join(data_path, "hotel_dict.txt")
encoding_file = os.path.join(data_path, "hotel_encoding.txt")

puncts = " \n"

with open(data_file, "r", encoding="utf-8-sig") as f:
    for line in f.readlines():
        trim_line = line.strip()
        for ch in trim_line:
            if ch in puncts:
                continue
            if ch in mydict:
                continue
            elif len(ch) <= 0:
                continue
            else:
                mydict[ch] = code
                code += 1
    code += 1
    mydict["<unk>"] = code

with open(dict_file, "w", encoding="utf-8-sig") as f:
    f.write(str(mydict))
    print("数据字典保存完成")


def load_dict():
    with open(dict_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return new_dict


# 对评论数据进行编码
new_dict = load_dict()

with open(data_file, "r", encoding="utf-8-sig") as f:
    with open(encoding_file, "w", encoding="utf-8-sig") as fw:
        for line in f.readlines():
            label = line[0]
            remark = line[2:-1]
            for ch in remark:
                if ch in puncts:
                    continue
                else:
                    fw.write(str(mydict[ch]))
                    fw.write(",")
            fw.write("\t" + str(label) + "\n")


def get_dict_len(dict_path):
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return len(new_dict.keys())


def data_mapper(sample):
    dt, lbl = sample
    val = [int(word) for word in dt.split(",") if word.isdigit()]
    return val, int(lbl)


def train_reader(train_list_path):
    def reader():
        with open(train_list_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            for line in lines:
                data, label = line.split("\t")
                yield data, label

        return paddle.reader.map_readers(
            data_mapper,  # 映射函数
            reader,  # 读取数据内容
            cpu_count(),  # 线程数量
            1024,  # 读取数据列大小
        )


def lstm_net(ipt, input_dim):
    ipt = fluid.layers.reshape(ipt, [-1, 1], inplace=True)
    # 词嵌入层
    emb = fluid.layers.embedding(
        input=ipt,
        size=[input_dim, 128],
        is_sparse=True
    )
    # 第一个全连接层
    fc1 = fluid.layers.fc(input=emb, size=128)

    # 第一分支:LSTM分支
    lstm1, _ = fluid.layers.dynamic_lstm(
        input=fc1,
        size=128
    )
    lstm2 = fluid.layers.sequence_pool(
        input=lstm1,
        pool_type='max'
    )
    # 第二分支
    conv = fluid.layers.sequence_pool(
        input=fc1,
        pool_type='max'
    )
    # 输出层：全连接层
    out = fluid.layers.fc(
        input=[conv, lstm2],
        size=2,
        act='softmax'
    )
    return out


# 定义输出数据，lod_level不为0指定输入数据为序列数据
dict_len = get_dict_len(dict_file)
rmk = fluid.layers.data(name="rmk", shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 定义长短期网络
model = lstm_net(rmk, dict_len)

# 定义损失函数，情绪判断实际是一个分类任务，使用交叉熵作为损失函数
cost = fluid.layers.cross_entropy(
    input=model,
    label=label
)
avg_cost = fluid.layers.mean(cost)  # 求损失值平均数

# layers.accuracy接口，用来评估预测准确率
acc = fluid.layers.accuracy(
    input=model,
    label=label
)

# 定义优化方法 Adagrad(自适应学习率，前期放大梯度调节，后期缩小梯度调节)
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.001)
opt = optimizer.minimize(avg_cost)
# 定义网络
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义reader
reader = train_reader(encoding_file)
batch_train_reader = paddle.batch(reader, batch_size=128)

# 定义输出数据的维度，数据的顺序是一条句子数据对应一个标签
feeder = fluid.DataFeeder(place=place, feed_list=[rmk, label])

for pass_id in range(40):
    for batch_id, data in enumerate(batch_train_reader):
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_cost, acc]
        )
        if batch_id % 20 == 0:
            print("pass_id:%d,batch_id:%d,cost:%0.5f,acc:%.5f" % (pass_id, batch_id, train_cost[0], train_acc))
        print("模型训练完成")

# 保存模型
model_save_dir = "model/chn_emotion_analyses.model"
if not os.path.exists(model_save_dir):
    print("create model path")
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(model_save_dir, feeded_var_names=[rmk.name], target_vars=[model], executor=exe)
print("模型保存完成，保存路径：", model_save_dir)

# 推理预测
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import random
from multiprocessing import cpu_count


def load_dict():
    with open(dict_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])

    return new_dict


def encode_by_dict(remark, dict_encoded):
    """
    根据字典对字符串进行编码
    :param remark:
    :param dict_encoded:
    :return:
    """
    remark = remark.strip()
    if len(remark) <= 0:
        return []
    ret = []
    for ch in remark:
        if ch in puncts:
            continue
        if ch in dict_encoded:
            ret.append(dict_encoded[ch])
        else:
            ret.append(dict_encoded["<unk>"])
    return ret


lods = []
new_dict = load_dict()
lods.append(encode_by_dict("", new_dict))

# 获取每句话的单词数量
base_shape = [len(c) for c in lods]

# 生成预测数据
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

tensor_words = fluid.create_lod_tensor(lods, base_shape, place)
infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(
    dirname=model_save_dir,
    executor=infer_exe
)

results = infer_exe.run(
    program=infer_program,
    feed={feed_target_names[0]: tensor_words},
    fetch_list=fetch_targets
)

# 打印每句话的正负面预测概率
for i, r in enumerate(results[0]):
    print("负面：%0.5f,正面：%0.5f" % (r[0], r[1]))
