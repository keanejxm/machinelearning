#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  05_email_class.py
:time  2024/3/13 17:38
:desc  
"""
import numpy as np
import re
import os
import string
import sklearn.model_selection as ms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from common_utils import DATA_PATH

label_name_map = ["垃圾邮件", "正常邮件"]


class EmailClass:
    def __init__(self):
        self.data_path = os.path.join(DATA_PATH, "nlp_data")

    @staticmethod
    def wipe_out_characters(text):
        """"""
        tokens = jieba.cut(text)
        tokens = [token.strip() for token in tokens]
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub("", token) for token in tokens])
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def wipe_out_word(self, content: str):
        """
        去除停用词
        :return:
        """
        stopword_path = os.path.join(self.data_path, "stop_words.utf8")
        with open(stopword_path, "r", encoding="utf8") as f:
            stopwords = [word for word in f.readlines()]
        tokens = jieba.cut(content)
        tokens = [token.strip() for token in tokens if token not in stopwords]
        words = " ".join(tokens)
        return words

    def read_spam_ham_data(self):
        """"""
        # 正常邮件
        ham_path = os.path.join(self.data_path, "ham_data.txt")
        # 垃圾邮件
        spam_path = os.path.join(self.data_path, "spam_data.txt")
        corpus = []
        labels = []
        with open(ham_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                corpus.append(line)
                labels.append(1)
        with open(spam_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                corpus.append(line)
                labels.append(0)
        return corpus, labels

    @staticmethod
    def remove_empty_docs(corpus, labels):
        """"""
        filtered_corpus = []
        filtered_labels = []
        for doc, label in zip(corpus, labels):
            if doc.strip():
                filtered_corpus.append(doc)
                filtered_labels.append(label)
        return filtered_corpus, filtered_labels

    def normalize_corpus(self, corpus):
        result = []
        for text in corpus:
            # 去除标点符号，去掉停用词
            text = self.wipe_out_characters(text)
            text = self.wipe_out_word(text)
            result.append(text)
        return result

    @staticmethod
    def tfidf_extractor(corpus):
        vectorzier = TfidfVectorizer(
            min_df=1,
            norm='l2',
            smooth_idf=True,
            use_idf=True
        )
        features = vectorzier.fit_transform(corpus)
        return vectorzier, features

    def print_metrics(self, true_labels, predicted_labels):
        """"""
        accuracy = metrics.accuracy_score(true_labels, predicted_labels)

        precision = metrics.precision_score(true_labels, predicted_labels, average="weighted")

        recall = metrics.recall_score(true_labels, predicted_labels, average="weighted")

        f1 = metrics.f1_score(true_labels, predicted_labels, average="weighted")
        print("正确率：%.2f，查准率：%.2f，召回率：%.2f，F1：%.2f" % (accuracy, precision, recall, f1))

    def start(self):
        corpus, labels = self.read_spam_ham_data()
        corpus, labels = self.remove_empty_docs(corpus, labels)
        print("总的数据量：", len(labels))
        for i in range(10):
            print("label:", labels[i], "邮件内容：", corpus[i])

        # 对数据划分训练集测试集
        train_corpus, test_corpus, train_labels, test_labels = ms.train_test_split(corpus, labels, test_size=0.10)
        # 规范化处理
        norm_train_corpus = self.normalize_corpus(train_corpus)
        norm_test_corpus = self.normalize_corpus(test_corpus)
        # 计算tf-idf
        tfidf_vectorizer, tfidf_train_features = self.tfidf_extractor(norm_train_corpus)
        # 测试集
        tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

        nb_model = MultinomialNB()  # 多分类朴素贝叶斯
        nb_model.fit(tfidf_train_features, train_labels)  # 训练

        mnb_pred = nb_model.predict(tfidf_test_features)  # 预测

        self.print_metrics(true_labels=test_labels, predicted_labels=mnb_pred)

        # 支持向量机
        svm_model = SGDClassifier()
        svm_model.fit(tfidf_train_features, train_labels)  # 训练
        svm_pred = svm_model.predict(tfidf_test_features)  # 预测

        self.print_metrics(true_labels=test_labels, predicted_labels=svm_pred)

        num = 0
        for text, label, pred_lbl in zip(test_corpus, test_labels, svm_pred):
            print("真实类别：", label_name_map[int(label)], "预测结果：", label_name_map[int(pred_lbl)])
            print("邮件内容", text.replace("\n", ""))
            num += 1
            if num == 10:
                break


if __name__ == '__main__':
    obj = EmailClass()
    obj.start()
