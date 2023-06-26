#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  stopwords.py
:time  2023/6/26 14:18
:desc  处理停用词
"""
import pandas as pd
import csv
DATA_PATH = r"E:\keane_data\nlp_data"

stopwords = pd.read_csv(f"{DATA_PATH}/stopwords.txt",header = None,quoting=csv.QUOTE_NONE,delimiter="\n")
stopwords_other = pd.read_csv(f"{DATA_PATH}/cn_stopwords.txt",header=None,quoting=csv.QUOTE_NONE,delimiter="\n")
stopwords = stopwords[0].tolist()
stopwords_other = stopwords_other[0].tolist()
other_stopwords = [stopword for stopword in stopwords_other if stopword not in stopwords]
new_stopwords = other_stopwords+stopwords
stopwords_df = pd.DataFrame(new_stopwords)
writer = pd.ExcelWriter(f"{DATA_PATH}/stopwords1.csv",engine="xlrd")
stopwords_df.to_csv(writer,index=False)
writer.save()
writer.close()