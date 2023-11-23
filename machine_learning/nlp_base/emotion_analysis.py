#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  emotion_analysis.py
:time  2023/11/20 15:09
:desc  文本情感分析
"""
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np
import os
import random
from multiprocessing import cpu_count

mydict = {}  # key：字 value:编码
code = 1
data_file = "hotel_discuss2.csv"  # 原始样本路径
encoding_file = "hotel_encoding.txt"  # 编码后的数据集
dict_file = "hotel_dict.txt"  # 字典文件
encoding_type = "utf-8-sig"  # 文件编码格式
puncts = " \n"  # 需要过滤的符号列表
# 数据集预处理
with open(data_file, "r", encoding=encoding_type) as f:
    for ln in f.readlines():
        trim_ln = ln.strip()
        for ch in ln:
            if ch in puncts:
                continue
            if ch in mydict:
                continue
            else:
                mydict[ch] = code
                code += 1
    mydict["<unk>"] = code

# 编码结束后将编码结果存入文件
with open(dict_file, "w", encoding=encoding_type) as f:
    f.write(str(mydict))
    print("保存字典文件完成")


# 读取字典中的文件
def load_dict():
    with open(dict_file, "r", encoding=encoding_type) as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
        return new_dict


new_dict = load_dict()

with open(data_file, "r", encoding=encoding_type) as f:
    with open(encoding_file, "w", encoding=encoding_type) as fw:
        for ln in f.readlines():
            label = ln[0]  # 第一个字符是标签
            remark = ln[2:]  # 评论部分
            for ch in remark:
                if ch in puncts:
                    continue
                else:
                    fw.write(str(new_dict[ch]))
                    fw.write(",")
            fw.write("\t" + str(label) + "\n")  # 写入tab,标签，类别
print("数据预处理完成")


# 模型定义与训练
# 获取字典长度

def get_dict_len(dict_path):
    with open(dict_path, "r", encoding=encoding_type) as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return len(new_dict.keys())


# 读取器

def data_mapper(sample):
    dt, label = sample
    val = [int(w) for w in dt.split(",") if w.isdigit()]
    return val, int(label)


def train_reader(train_path):
    def reader():
        with open(train_path, "r", encoding=encoding_type) as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            for ln in lines:
                data, label = ln.split("\t")
                yield data, label

    return paddle.reader.xmap_readers(
        data_mapper,
        reader,
        cpu_count(),
        1024  # 缓冲区大小
    )

# 定义模型
def lstm_net(input,input_dim):
    input = fluid.layers.reshape(input,[-1,1],inplace = True)
    # 词嵌入层
    emb = fluid.layers.embedding(input = input,
                                 size = [input_dim,128],# 词嵌入
                                 is_sparse=True
                                 )
    # 第一个全连接层
    fc1 = fluid.layers.fc(input = emb,size = 128)
    # 第一个分支
    lstm1,_ = fluid.layers.dynamic_lstm(input = fc1,size = 128)
    lstm2 = fluid.layers.sequence_pool(input = lstm1,pool_type = "max")

    # 第二个分支
    conv = fluid.layers.sequence_pool(input =fc1,)

    # 输出层
    out= fluid.layers
    return out

dict_len = get_dict_len(dict_file)
rmk = fluid.layers.data(name = "rmk",shape = [1],dtype = "int64",lod_level = 1)
label = fluid
model = lstm_net(rmk,dict_len)
# 损失函数
cost= fluid.layers.cross_entropy(input = model,label = label)
# 测试
