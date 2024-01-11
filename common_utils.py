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
print(DATA_PATH)
