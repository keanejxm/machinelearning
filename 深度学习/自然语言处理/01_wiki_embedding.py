#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 19:53
# @Author  : keane
# @Site    : 
# @File    : 01_wiki_embedding.py
# @Software: PyCharm
# 使用维基百科中文语料库训练词向量
import logging
import os
import os.path
from gensim.corpora import WikiCorpus

# 获取输入数据
input_file = "articles.xml.bz2"

# 输出文件
out_file = open("wiki.zh.text", "w", encoding="utf8")

count = 0  # 控制读取多少样本

wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})

for text in wiki.get_texts():
    out_file.write(" ".join(text) + "\n")
    count += 1

    if count % 200 == 0:
        print("解压笔数", count)

    if count >= 20000:
        break

out_file.close()

# 2.分词
import jieba
import jieba.analyse
import codecs


def process_wiki_text(src_file, dest_file):
    with codecs.open(src_file, "r", "utf8") as f_in, codecs.open(dest_file, "w", encoding="utf8") as f_out:
        num = 1
        for ln in f_in.readlines():  # 读取每行
            line_seg = " ".join(jieba.cut(ln))  # 分词
            f_out.writelines(line_seg)  # 写到输出文件中
            num += 1
            if num % 200 == 0:
                print("分词完成笔数:", num)


process_wiki_text("wiki.zh.text", "wiki.zh.text.seg")

# 3.训练词向量
import logging
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence # 按行读取
logger = logging.getLogger(__name__)
# format: 指定输出的格式和内容，format可以输出很多有用信息，
# %(asctime)s: 打印日志的时间
# %(levelname)s: 打印日志级别名称
# %(message)s: 打印日志信息
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


in_file = "wiki.zh.text.seg" # 输入文件(经过分词后的)
out_file1 = "wiki.zh.text.model" # 模型
out_file2 = "wiki.zh.text.vector" # 权重


model = Word2Vec(LineSentence(in_file), # 输入
                 vector_size=100, # 词向量维度(推荐25~300之间)
                 window=3, # 窗口大小
                 min_count=5, # 如果语料中单词出现次数小于5，忽略该词
                 workers=multiprocessing.cpu_count()) # 线程数量

# 保存模型
model.save(out_file1)
# 保存权重矩阵C
model.wv.save_word2vec_format(out_file2, # 文件路径
                              binary=False) # 不保存二进制

# 4.测试
import gensim
from gensim.models import Word2Vec

# 加载模型
model = Word2Vec.load("wiki.zh.text.model")
count = 0

# for word in model.wv.index2word:
#     print(word, model[word]) # 打印
#     count += 1
#     if count >= 10:
#         break

# print("==================================")

result = model.wv.most_similar(u"铁路")

for r in result:
    print(r)

print("==================================")

result = model.wv.most_similar(u"中药")
for r in result:
    print(r)