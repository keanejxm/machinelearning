#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  matplotlib_demo2.py
:time  2023/12/1 13:57
:desc  
"""
import pandas as pd
import matplotlib.pyplot as plt
from common_utils import *
data = pd.read_csv(f"{DATA_PATH}/Salary_Data.csv")
year_experience = data["YearsExperience"]
salary = data["Salary"]

plt.scatter(year_experience,salary,c = salary)
plt.show()