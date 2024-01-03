#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 14:56
# @Author  : keane
# @Site    : 
# @File    : 01_jieba_cut_demo.py
# @Software: PyCharm
import jieba

text = "吉林市长春药店"

# 精确模式
seg_list = jieba.cut(text,
                     # cut_all=False)# 是否做全模式分词
                     cut_all=True)  # 全模式分词
# 结果为生成器
for word in seg_list:
    print(word, end="/")

print("")

# 搜索引擎模式

seg_list = jieba.cut_for_search(text)
for word in seg_list:
    print(word,end="/")
