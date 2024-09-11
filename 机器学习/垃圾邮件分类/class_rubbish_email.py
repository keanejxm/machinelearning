#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 21:51
# @Author  : keane
# @Site    : 垃圾邮件分类ham:正常，spam:垃圾邮件
# @File    : class_rubbish_email.py
# @Software: PyCharm
import os
import re

import jieba
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
import sklearn.metrics as sm


class ClassRubbishEmail:
    def __init__(self):
        if os.path.exists(r"D:\keane_data\02_nlp_data"):
            self.nlp_dir = r"D:\keane_data\02_nlp_data"
        else:
            self.nlp_dir = r"E:\keane_data\nlp_data"
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

    # 删掉空数据
    def clear_data(self, corpus, labels):
        filter_corpus = list()
        filter_label = list()
        for data, label in zip(corpus, labels):
            # 删除掉空数据
            if not data:
                continue
            # 对数据进行处理去掉标点符号和停用词

            # 分词
            data_cut: list = jieba.lcut(data)
            # 删除标点符号
            data_symbol = self.filter_symbol(data_cut)
            # 去除停用词
            data_stopword: list = [word for word in data_symbol if word not in self.stopwords_list]
            new_data = "".join(data_stopword)
            filter_corpus.append(new_data)
            filter_label.append(label)
        return filter_corpus, filter_label

    @staticmethod
    def filter_symbol(datas: list):
        """"""
        puns = string.punctuation
        pattern = re.escape(puns)
        new_data = [re.sub(pattern, "", data) for data in datas]
        return new_data

    def tf_idf_method(self, corpus):
        vector = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True)
        tf_idf_res = vector.fit_transform(corpus)
        return tf_idf_res, vector

    # 构建模型
    @staticmethod
    def train_model(x, y):
        """"""
        model = nb.MultinomialNB()
        model.fit(x, y)
        return model

    def start(self):
        """
        1.读取数据
        2.清洗数据：去掉空数据，去掉标点符号，去掉停用词
        :return:
        """
        corpus, labels = self.read_email()
        # 对邮件数据进行清洗
        corpus, labels = self.clear_data(corpus, labels)
        # 训练集，测试集
        train_x, test_x, train_y, test_y = train_test_split(corpus, labels, test_size=0.1)
        # tfidf处理文字
        train_x, vector = self.tf_idf_method(train_x)
        model = self.train_model(train_x, train_y)
        test_x = vector.transform(test_x)
        pred_y = model.predict(test_x)
        # 模型评估
        res_sm = sm.classification_report(y_true=test_y, y_pred=pred_y)
        print(res_sm)


if __name__ == '__main__':
    cre_obj = ClassRubbishEmail()
    cre_obj.start()
