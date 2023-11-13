#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  o1_stopword_demo.py
:time  2023/11/9 16:06
:desc  
"""
import jieba


def get_stopword_list():
    stop_word_path = "stopword.txt"
    with open(stop_word_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        stopword_list = [line.strip() for line in lines]
        return stopword_list


print(get_stopword_list())
