#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  deal_sohu_news.py
:time  2023/6/26 15:52
:desc  
"""
import json
import re

DATA_PATH = r"E:\keane_data\nlp_data\nlp_project\nlp_code"
classify_map = {
    "http://www.xinhuanet.com/auto/": "汽车",
"http://www.xinhuanet.com/fortune":"财经",
"http://www.xinhuanet.com/internet/":"IT",
"http://www.xinhuanet.com/health/":"健康",
"http://www.xinhuanet.com/sports":"体育",
"http://www.xinhuanet.com/travel":"旅游",
"http://www.xinhuanet.com/edu":"教育",
"http://www.xinhuanet.com/employment":"招聘",
"http://www.xinhuanet.com/life":"文化",
"http://www.xinhuanet.com/mil":"军事",
"http://www.xinhuanet.com/olympics/":"奥运",
"http://www.xinhuanet.com/society":"社会"
}
classify_list = list()
with open(f"{DATA_PATH}/news_sohusite_xml.dat", "r",encoding="gb18030") as f:
    data = dict()
    num = 1
    strip_data = dict()
    for line in f:
        if line.startswith("<doc>"):
            strip_data = dict()
        elif line.startswith("<url>"):
            url = line.strip().lstrip("<url>").rstrip("</url>")
            classify = re.findall(r"http://(\S+)\.sohu",url)
            if classify:
                strip_data["classify"] = classify[0]
                if classify[0] not in classify_list:
                    classify_list.append(classify[0])
            else:
                strip_data["classify"] = ""
                print(url)
            strip_data["url"] = url
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
print(classify_list)