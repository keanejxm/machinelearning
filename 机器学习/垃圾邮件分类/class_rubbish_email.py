#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 21:51
# @Author  : keane
# @Site    : 垃圾邮件分类ham:正常，spam:垃圾邮件
# @File    : class_rubbish_email.py
# @Software: PyCharm
import os
import jieba
import string
from sklearn.feature_extraction.text import TfidfVectorizer


class ClassRubbishEmail:
    def __init__(self):
        self.nlp_dir = r"D:\keane_data\02_nlp_data"
        self.rubbish_dir = os.path.join(self.nlp_dir, "rubbish_email_data")
        self.ham_path = os.path.join(self.rubbish_dir, "ham_data.txt")
        self.spam_path = os.path.join(self.rubbish_dir, "spam_data.txt")
        self.stopwords_path = os.path.join(self.nlp_dir, "stopword.txt")
        self.stopwords_list = self.read_stopwords()

    # 读取停用词
    def read_stopwords(self):
        stopwords_list = list()
        with open(self.stopwords_path, "r", encoding="utf8") as r:
            for word in r.readlines():
                stopwords_list.append(word.strip())
        return stopwords_list

    # 读取正常和垃圾邮件
    def read_email(self):
        """"""
        corpus = list()
        labels = list()
        with open(self.ham_path, "r", encoding="utf8") as r:
            for data in r.readlines():
                corpus.append(data.strip())
                labels.append(1)
        with open(self.spam_path, "r", encoding="utf8") as r:
            for data in r.readlines():
                corpus.append(data)
                labels.append(0)

        return corpus, labels

    def clear_space_data(self, corpus, labels):
        filter_corpus = list()
        filter_label = list()
        for data, label in zip(corpus, labels):
            if data:
                filter_corpus.append(data)
                filter_label.append(label)
        return filter_corpus, filter_label

    def participle(self, corpus, labels):
        cut_corpus = list()
        cut_labels = list()
        for data, label in zip(corpus, labels):
            cut_data = jieba.lcut(data)
            # 去除停用词
            cut_data = [word for word in cut_data if word not in self.stopwords_list]
            cut_corpus.append(cut_data)
            cut_labels.append(label)
        return cut_corpus, cut_labels

    def filter_symbol(self, corpus, labels):
        """"""
        new_corpus = list()
        new_labels = list()
        pun_list = [pun for pun in string.punctuation]
        for data, label in zip(corpus, labels):
            data = "".join([word for word in data if word not in pun_list])
            new_corpus.append(data)
            new_labels.append(label)
        return new_corpus, new_labels

    def tf_idf_method(self, corpus, labels):
        tf_idf = TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True)
        tf_idf.fit(corpus)
        tf_idf.transform(corpus)

    def start(self):
        corpus, labels = self.read_email()
        # 对邮件进行处理，去除空数据
        corpus, labels = self.clear_space_data(corpus, labels)
        # 对数据进行分词
        corpus, labels = self.participle(corpus, labels)
        # 去除符号
        corpus, labels = self.filter_symbol(corpus, labels)
        # 去除停用词
        print(corpus, labels)


if __name__ == '__main__':
    cre_obj = ClassRubbishEmail()
    cre_obj.start()
