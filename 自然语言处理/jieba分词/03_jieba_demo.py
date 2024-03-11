#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  03_jieba_demo.py
:time  2024/3/11 15:39
:desc  jieba词性标注
"""
import jieba.posseg as psg


def pos(text):
    results = psg.cut(text)
    for w, t in results:
        print("%s/%s" % (w, t), end=" ")
    print("")


text = "呼伦贝尔大草原"
pos(text)
text = "梅兰芳大剧院周六晚上有演出"
pos(text)
text = "哈哈哈哈"
pos(text)
