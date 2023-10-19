#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  test1.py
:time  2023/10/18 9:57
:desc  
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
fig = plt.figure()
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.show()
fig.savefig("sc.png")