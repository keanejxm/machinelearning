#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  word_2_vec_demo.py
:time  2023/6/25 15:39
:desc  
"""
import json
import os

import pandas as pd
import csv
import jieba
import gensim.models.word2vec
import multiprocessing
from keras.datasets import imdb
from lxml import etree
NLP_DATA_PATH = r"E:\keane_data\nlp_data\nlp_project\nlp_code"
DATA_PATH = r"E:\keane_data\nlp_data"
# 下载imdb数据集
# sentences = [["first","sentence"],["second","sentence"]]
# model = gensim.models.Word2Vec(sentences,min_count=1)
# print(model.wv["first"])
# cup_cores = multiprocessing.cpu_count()
# (train_data,train_labels),(test_data,test_labels) = imdb.load_data(path="imdb.npz",num_words=1000)
# print(train_data,train_labels)
# 处理文本

# sentence = pd.read_csv(f"{DATA_PATH}\love.txt",header = None,quoting=csv.QUOTE_NONE,delimiter="\n")
# sentence.columns = ["内容"]
# sentence["内容"] = sentence["内容"].str.replace("\（.*\）","")
# sentence = sentence["内容"].tolist()
# stopwords = pd.read_csv(f"{DATA_PATH}\cn_stopwords.txt",header=None,quoting=csv.QUOTE_NONE,delimiter="\n")
# stopwords = stopwords[0].tolist()
# stopwords.append("时")
# stopwords.append("一种")
# stopwords.append("请")
# stopwords.append("●")
# sentence_cut = [" ".join(jieba.lcut(line))for line in sentence]
# sentence_no_stopwords = [[word for word in line.split() if word not in stopwords] for line in sentence_cut]
# from collections import defaultdict
# frequency = defaultdict(int)
# for line in sentence_no_stopwords:
#     for token in line:
#         frequency[token] +=1
# sentence_morethan1time = [[token for token in line if frequency[token]>3] for line in sentence_no_stopwords]
# print(sentence_morethan1time)
#
# # 算法
# model = gensim.models.Word2Vec(sentence_morethan1time,min_count = 2,window= 5)
# print(model.wv.key_to_index)
# print(model.wv["爱情"])
# print(model.wv.similar_by_word("浪漫", topn=10, restrict_vocab=30))

# 读取搜狐新闻内容
with open(f"{NLP_DATA_PATH}/sohu_news.json","r",encoding="utf8") as r:
    data = json.load(r)
    r.close()
content_data = list()
stopwords = pd.read_csv(f"{DATA_PATH}/stopwords.txt",header= None,quoting=csv.QUOTE_NONE,delimiter="\n")
stopwords = stopwords[0].tolist()
for news_index,news_data in data.items():
    content = news_data["content"]
    # 去掉停用词
    content_stopwords = [word for word in jieba.lcut(content)if word not in stopwords and not word.startswith(r"\u")]
    content_data.append(content_stopwords)

# 训练模型
model = gensim.models.Word2Vec(content_data,window=5,min_count=2)
# 保存模型
if not os.path.exists("word2vec"):
    model.save("word2vec")
else:
    pass