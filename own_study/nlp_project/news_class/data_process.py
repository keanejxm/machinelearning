#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  data_process.py
:time  2023/1/17 14:15
:desc  数据预处理
"""
import jieba
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcess():
    def __init__(self, data_path, stopwords_path):
        self._data_path = data_path
        self._stopwords_path = stopwords_path

    def _read_data(self):
        """
        读取数据
        :return:
        """
        with open(self._data_path, "r", encoding="utf8") as r1:
            try:
                data = r1.readlines()
            finally:
                r1.close()
        with open(self._stopwords_path, "r", encoding="utf8") as r2:
            try:
                stopwords_original = r2.readlines()
            finally:
                r2.close()
        # 处理停用词
        stopwords = list()
        for word in stopwords_original:
            stopwords.append(word[0:-1])
        return data, stopwords

    def _deal_data(self, data, stopwords):
        """
        处理新闻数据
        :return:
        """
        x_data = list()
        y_data = list()
        for news_data in data:
            y_text, x_text = news_data.split("\t", 1)
            y_data.append(y_text)
            seg_text = [word for word in jieba.cut(x_text) if word not in stopwords]
            x_data.append(" ".join(seg_text))
        label_encoder = LabelEncoder()
        label_nums = label_encoder.fit_transform(y_data)
        label_nums = np.array([label_nums]).T
        categories = list(label_encoder.classes_)
        encoder_one_hot = OneHotEncoder()
        label_one_hot = encoder_one_hot.fit_transform(label_nums)
        label_one_hot = label_one_hot.toarray()
        return model_selection.train_test_split(x_data, label_one_hot, test_size=0.2, random_state=1024)

    def _get_tfidf(self, x_train, x_test):
        """
        提取tf-idf特征
        :param x_train:
        :param x_test:
        :return:
        """
        vectorizer = TfidfVectorizer(min_df=100)
        vectorizer.fit_transform(x_train)
        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)
        return x_train_vec, x_test_vec, vectorizer

    def start(self):
        """
        程序开始
        :return:
        """
        data, stopwords = self._read_data()
        x_train, x_test, y_train, y_test = self._deal_data(data, stopwords)
        x_train_vec, x_test_vec, vectorizer = self._get_tfidf(x_train, x_test)
        print("lllllllllll")


if __name__ == '__main__':
    data_path = r"E:\keane\github\machinelearning\nlp_sss\news_class\data\cnews\cnews.train.txt"
    stopwords_path = r"E:\keane\github\machinelearning\nlp_sss\news_class\data\stopwords.txt"
    obj = DataProcess(data_path, stopwords_path)
    obj.start()
