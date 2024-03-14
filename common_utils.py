#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  common_utils.py
:time  2023/12/1 9:33
:desc  
"""
import os

data_path_1 = r"E:\keane_data"

if os.path.exists(data_path_1):
    DATA_PATH = data_path_1
else:
    DATA_PATH = r"D:\keane_data"

WORD_MAP = {
    "ag": "形容素",  # 形容词性语素
    "a": "形容词",
    "ad": "副形词",
    "an": "名形词",
    "b": "区别词",
    "c": "连词",
    "dg": "副语素",
    "d": "副词",
    "e": "叹词",
    "f": "方位词",
    "g": "语素",
    "h": "前接成分",
    "i": "成语",
    "j": "简称略语",
    "k": "后接成分",
    "l": "习用语",
    "m": "数词",
    "ng": "名语素",
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "nt": "机构团体",
    "nz": "其他专名",
    "o": "拟声词",
    "p": "介词",
    "q": "量词",
    "r": "代词",
    "s": "处所词",
    "tg": "时语素",
    "t": "时间词",
    "u": "助词",
    "vg": "动语素",
    "v": "动词",
    "vd": "副动词",
    "vn": "名动词",
    "w": "标点符号",
    "x": "非语素字",
    "y": "语气词",
    "z": "状态词",
}
