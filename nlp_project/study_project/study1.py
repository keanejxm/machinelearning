#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  study1.py
:time  2022/12/23 14:57
:desc  
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# x, y = datasets.make_classification(n_samples=1000, n_features=100, n_redundant=0, random_state=1)
# train_X, test_X, train_Y, test_y = train_test_split(x, y)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(train_X, train_Y)
# pred_Y = knn.predict(test_X)
# print(metrics.confusion_matrix(test_y, pred_Y))
# print(metrics.recall_score(test_y, pred_Y))
# print(metrics.precision_score(test_y, pred_Y))
# print(metrics.accuracy_score(test_y, pred_Y))
# print(metrics.f1_score(test_y, pred_Y))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
print(vectorizer)
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
x = vectorizer.fit_transform(corpus)
print(x.toarray())
bb = vectorizer.get_feature_names()
aa = vectorizer.get_feature_names_out()
print(aa)
print(x.toarray())
vocabulary = vectorizer.vocabulary_
print(vocabulary)
new_vectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary)
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]
          ]
tfidf = transformer.fit_transform(x)
print(tfidf.toarray())

import tensorflow as tf

x_text = [
    'i love you',
    'me too'
]

aa = ["今天 太阳 真大",
      "明天 太阳 真大",
      "今天 吃饭 好饱"]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
bb = tfidf_vectorizer.fit_transform(aa)
print(bb.toarray())
print(tfidf_vectorizer.vocabulary_)
print(tfidf_vectorizer.get_feature_names())
print("---------------")

aaa = ["a", "b", "c"]

bbb = "abcd"

ccc = list()
ddd = [word not in aaa for word in bbb]
eee = [word for word in bbb if word not in aaa]
print(ddd)
print(eee)
