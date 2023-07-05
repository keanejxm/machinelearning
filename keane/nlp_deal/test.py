#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  test.py
:time  2023/6/30 9:16
:desc  
"""
# from nltk.tokenize import TreebankWordTokenizer
# doc_0 = "The faster Harry got to the store, the faster Harry, the faster, would get home."
# doc_1 = "Harry is hairy and faster than Jill."
# doc_2 = "Jill is not as hairy as Harry."
# docs = [doc_0,doc_1,doc_2]
# doc_tokens = list()
# tokenier = TreebankWordTokenizer()
# for doc in docs:
#     doc_tokens += [sorted(tokenier.tokenize(doc.lower()))]
# all_doc_tokens = sum(doc_tokens,[])
# lexicon = sorted((set(all_doc_tokens)))
# print(lexicon)
# from collections import OrderedDict
# zero_vector = OrderedDict((token,0) for token in lexicon)
# import copy
# doc_voctors = list()
# from collections import Counter
# for doc in docs:
#     vec = copy.copy(zero_vector)
#     tokens = tokenier.tokenize(doc.lower())
#     token_counts = Counter(tokens)
#     for key,value in token_counts.items():
#         vec[key] = value/len(lexicon)
#     doc_voctors.append(vec)
# import pandas as pd
# bb = pd.DataFrame(doc_voctors)
# print(bb)


# -----------------词频背后的意义---------------
# import numpy as np
#
# topic = {}
# tfidf = dict(list(zip("cat dog apple lion NYC love".split(), np.random.rand(6))))
# topic["petness"] = ()
# a = [
#     [.3, .3, 0, 0, 2, 2],
#     [.1, .1, .1, .5, .1, .1],
#     [0, .1, .2, .1, .5, .1]
# ]
# b = ["cat", "dog", "apple", "lion", "NYC", "love"]
#
# c = ["p","l","c"]
# c = a*b\


# 一个LDA分类器
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize.casual import casual_tokenize
import csv

# sms = pd.read_csv("sms-spam.csv")
# sms["spam"] = sms.spam.astype(int)
# # tokenizer 指定分词器中文jieba.lcut
# tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
# tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text)
# tfidf_docs = tfidf_docs.toarray()
# print(tfidf_docs.shape)
# voc_ = tfidf_model.vocabulary_
# voc_ = sorted(voc_.items(), key=lambda x:x[1])
# mask = sms.spam.astype(bool).values
# spam_centroid = tfidf_docs[mask].mean(axis=0)
# ham_centroid = tfidf_docs[~mask].mean(axis=0)
# a = spam_centroid-ham_centroid
# spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
# b = spamminess_score.reshape(-1, 1)
# from sklearn.preprocessing import MinMaxScaler
#
# sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1, 1))
# sms['lda_predict'] = (sms.lda_score > .5).astype(int)
# print(sms['spam lda_predict lda_score'.split()].round(2).head(6))

aa = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
u,s,vt = np.linalg.svd(aa)
print(vt)
