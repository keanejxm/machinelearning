#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  dataset_demo.py
:time  2023/3/6 11:16
:desc  
"""
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris["data"]
print(iris_data)