#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 15:12
# @Author  : keane
# @Site    : 
# @File    : 02_term_frequency_demo.py
# @Software: PyCharm

import jieba


# 读取文件内容

def get_content(path):
    with open(path, "r", encoding="gbk") as f:
        content = "".join([ln.strip() for ln in f.readlines()])
    return content


# 统计词语出现的次数
def get_tf(words, topk=10):
    """
    words:分词后的列表
    topk:打印前k个词的数量
    """
    tf_dict = {}
    for w in words:
        if w not in tf_dict:
            tf_dict[w] = 1
        else:
            tf_dict[w] += 1
    # 按出现次数倒序排序
    sort_list = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)
    return sort_list[0:topk]


# 获取停用词表
def get_stop_words(path):
    with open(path, "r", encoding="utf8") as f:
        stop_words = [ln.strip() for ln in f.readlines()]
    return stop_words


if __name__ == '__main__':
    fname = "11.txt"
    corpus = get_content(fname)
    tmp_list = list(jieba.cut(corpus))
    tmp_list1 = jieba.lcut(corpus)
    print(tmp_list1)
    print(tmp_list)
    stop_words = get_stop_words("stop_words.utf8")
    # 过滤停用词
    split_words = []
    for w in tmp_list:
        if w not in stop_words:
            split_words.append(w)

    split_words_1 = [w for w in tmp_list if w not in stop_words]
    print(split_words)
    print(split_words_1)
    print(get_tf(split_words))