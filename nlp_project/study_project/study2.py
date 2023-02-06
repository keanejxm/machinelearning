#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  study2.py
:time  2023/1/31 9:08
:desc  
"""
import os
import jieba
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


class Study2():
    def __init__(self):
        self._file_path = f"{os.path.dirname(__file__)}"

    @staticmethod
    def _read_data_txt(file_path):
        """
        读取txt数据样本
        :return:
        """
        with open(file_path, "r", encoding="utf8") as r:
            try:
                data = r.readlines()
                new_data = list()
                for word in data:
                    new_data.append(word.rstrip("\n"))
            finally:
                r.close()
        return new_data

    def _deal_data(self, data, stopwords):
        """
        处理数据
        :return:
        """
        y_list = list()
        title_list = list()
        abstract_list = list()
        num = 0
        for row_data in data:
            num += 1
            data_list = row_data.rstrip("\n").split("_!_")
            class_name = data_list[2]  # 类名
            news_title = data_list[3]  # 新闻标题
            news_abstract = data_list[4]  # 新闻简介
            news_title_cut = [word for word in jieba.lcut(news_title) if word not in stopwords]
            news_abstract_cut = [word for word in jieba.lcut(news_abstract) if word not in stopwords]
            if class_name and news_title_cut and news_abstract_cut:
                title_list.append(" ".join(news_title_cut))
                abstract_list.append(" ".join(news_abstract_cut))
                y_list.append(class_name)
                print(f"正在解析第{num}条数据")
        return title_list, abstract_list, y_list

    def _data_translate_features(self):
        """
        将数据转为特征
        :return:
        """

    def _tf_idf_1(self, data):
        """
        第一种tf-idf处理数据
        :return:
        """
        count_vec = CountVectorizer()
        features = count_vec.fit_transform(data)
        # print(features.toarray())
        # print(count_vec.vocabulary_)
        # print(count_vec.get_feature_names())
        tfidf_transform = TfidfTransformer()
        tfidf_data = tfidf_transform.fit_transform(features)
        bb = tfidf_data.toarray()
        print(tfidf_data.toarray())
        bb = tfidf_data.toarray()

    def _tf_idf_2(self, data):
        """
        第二种tf-idf处理数据
        :return:
        """
        tfidf_vec = TfidfVectorizer()
        tfidf_data = tfidf_vec.fit_transform(data)
        bb = tfidf_data.toarray()
        print(tfidf_data.toarray())

    def _deal_y_data(self, data):
        """
        处理目标向量，labelEncoder,OneHotEncoder
        :param data:
        :return:
        """
        label_encoder = LabelEncoder()
        label_nums = label_encoder.fit_transform(data)
        categories = label_encoder.classes_
        label_nums = np.array([label_nums]).T
        one_hot_encoder = OneHotEncoder()
        label_one_hot=one_hot_encoder.fit_transform(label_nums)
        bb = label_one_hot.toarray()
        print(bb)
        print(categories)

    def start(self):
        """
        程序开始入口
        :return:
        """
        data_path = f"{self._file_path}/study_data/toutiao_cat_data.txt"
        stopwords_path = f"{self._file_path}/study_data/stopwords.txt"
        data = self._read_data_txt(data_path)
        stopwords_data = self._read_data_txt(stopwords_path)
        title_data, abstract_data, y_data = self._deal_data(data, stopwords_data)

        self._deal_y_data(y_data)
        self._tf_idf_1(abstract_data)
        self._tf_idf_2(abstract_data)

        # print(title_data)
        # print(abstract_data)
        # print(y_data)


if __name__ == '__main__':
    obj = Study2()
    obj.start()
