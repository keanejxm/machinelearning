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
# x = np.linspace(0,10,100)
# fig = plt.figure()
# plt.plot(x,np.sin(x))
# plt.plot(x,np.cos(x))
# plt.show()
# fig.savefig("sc.png")

arr1 = [1, 3, False, 4]
arr2 = [3, 0, True, False]

# output
out_arr = np.logical_and(arr1, arr2)

print("Output Array : ", out_arr)

a = np.array([1,2,3,4])
print(a[:2])