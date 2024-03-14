#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  01_jieba_demo.py
:time  2024/3/11 15:01
:desc  
"""
import os
import jieba
import jieba.posseg as psg
from common_utils import DATA_PATH
from common_utils import WORD_MAP

data_path = os.path.join(DATA_PATH, "nlp_data")

file_path = os.path.join(data_path, "11.txt")
with open(file_path, "r", encoding="GBK") as f:
    content = ""
    for line in f.readlines():
        content += (line.strip())

# 读取停用词
stopword_file_path = os.path.join(data_path, "stop_words.utf8")
with open(stopword_file_path, "r", encoding="utf8") as r:
    stopwords = list()
    for word in r.readlines():
        stopwords.append(word.strip())
# 去除停用词
new_content = ""
for word in content:
    if word in stopwords:
        continue
    else:
        new_content += word
# jieba分词：全模式、精确模式、搜索模式
seg_list = jieba.lcut(new_content, cut_all=True)
print("全模式:", seg_list)
seg_list = jieba.lcut(new_content)
print("精确模式:", seg_list)
seg_list = jieba.lcut(new_content, cut_all=False)
print("精确模式：", seg_list)
seg_list = jieba.lcut_for_search(new_content)
print("搜索引擎模式：", seg_list)

results = psg.cut(new_content)
for w, t in results:
    print(w, WORD_MAP[t])
