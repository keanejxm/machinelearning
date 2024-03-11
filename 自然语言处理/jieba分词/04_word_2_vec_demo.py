#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  04_word_2_vec_demo.py
:time  2024/3/11 17:09
:desc  
"""
from gensim.corpora import WikiCorpus

# 1、解压数据集
input_file = ""
# 输出文件
out_file = open("", "w", encoding="utf8")
count = 0  # 控制读取多少笔样本
wiki = WikiCorpus(
    input_file,
    lemmatize=False,  # 时态
    dictionary={})
for text in wiki.get_texts():  # 逐笔读取数据
    out_file.write(" ".join(text) + "\n")  # 写入输出文件
    count += 1
    if count % 200 == 0:
        print("解压笔数：", count)
    if count >= 20000:
        break
out_file.close()
# 2、分词、
import jieba
import jieba.analyse
import codecs


def process_wiki_text(src_file, dest_file):
    with codecs.open(src_file, "r", "utf-8") as f_in, \
            codecs.open(dest_file, "w", "utf-8") as f_out:
        num = 1
        for ln in f_in.readlines():
            line_seg = " ".join(jieba.cut(ln))
            f_out.write(line_seg)
            num += 1
            if num % 200 == 0:
                print("分词完成笔数：", num)


process_wiki_text("wiki.zh.text", "wiki.zh.text.seg")

# 3、训练词向量
import logging
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence  # 按行读取
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=""
)
logging.root.setLevel(level=logging.INFO)
in_file = ""
out_file1 = ""
out_file2 = ""
# 定义模型
model = Word2Vec(
    LineSentence(in_file),
    vector_size = 100,# 词向量维度
    window=3,# 窗口大小
    min_count=5,
    workers=multiprocessing.cpu_count()
)
# 保存模型
model.save(out_file1)
# 保存权重
model.wv.save_word2vec_format(out_file2,binary=False)
# 4、测试
# 加载模型
model =Word2Vec.load("wiki.zh.text.model")
count = 0
for word in model.wv.index2word:
    print(word,model[word])
    count +=1
    if count>=10:
        break

result = model.wv.most_similar(u"铁路")
result = model.wv.most_similar(u"中药")
