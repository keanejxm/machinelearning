#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  sigmoid.py
:time  2023/11/2 17:19
:desc  
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
res = 1/(1+np.e**-x)

plt.plot(x,res)
plt.show()