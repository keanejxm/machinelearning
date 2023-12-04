#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pandas_demo2.py
:time  2023/12/1 9:30
:desc  
"""
import pandas as pd
import numpy as np
from common_utils import *

data = pd.read_json(f"{DATA_PATH}/ratings.json")
print(data)

fracture = data.loc["Fracture"]
print(np.mean(fracture))
