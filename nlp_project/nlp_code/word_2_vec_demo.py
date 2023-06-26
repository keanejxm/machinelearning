#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  word_2_vec_demo.py
:time  2023/6/25 15:39
:desc  
"""
import gensim
import multiprocessing
from keras.datasets import imdb
# 下载imdb数据集
sentences = [["first","sentence"],["second","sentence"]]
model = gensim.models.Word2Vec(sentences,min_count=1)
print(model.wv["first"])
# cup_cores = multiprocessing.cpu_count()
# (train_data,train_labels),(test_data,test_labels) = imdb.load_data(path="imdb.npz",num_words=1000)
# print(train_data,train_labels)
