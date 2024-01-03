#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 16:30
# @Author  : keane
# @Site    : 
# @File    : 03_pos_jieba_demo.py
# @Software: PyCharm


import jieba.posseg as psg


def pos(text):
    result = psg.cut(text)  # 分词词性标注
    for w, t in result:
        print("%s/%s" % (w, t))

text = "呼伦贝尔大草原"

pos(text)

text = "梅兰芳大剧院周六晚上有演出"
pos(text)