#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  deal_sohu_news.py
:time  2023/6/26 15:52
:desc  
"""
import json

DATA_PATH = r"E:\keane_data\nlp_data\nlp_project\nlp_code"
with open(f"{DATA_PATH}/news_sohusite_xml.dat", "r",encoding="gb18030") as f:
    data = dict()
    num = 1
    strip_data = dict()
    for line in f:
        if line.startswith("<doc>"):
            strip_data = dict()
        elif line.startswith("<contenttitle>"):
            strip_data["title"] = line.strip().lstrip("<contenttitle>").rstrip("</contenttitle>")
        elif line.startswith("<content>"):
            strip_data["content"] =line.strip().lstrip("<content>").rstrip("</content>")
        elif line.startswith("</doc>"):
            data[str(num)] = strip_data
            print(num)
            num +=1
        else:
            continue
    with open(f"{DATA_PATH}/sohu_news.json","w",encoding="UTF8") as w:
        w.write(json.dumps(data,ensure_ascii=False))
        w.close()