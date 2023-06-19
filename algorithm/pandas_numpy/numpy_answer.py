#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  numpy.py
:time  2023/6/19 15:52
:desc  
"""
# 练习
# 打印numpy的版本和配置说明
import numpy as np

print(np.__version__)
np.show_config()
# 输出：
#
# 1.21.3
# blas_mkl_info:
#   NOT AVAILABLE
# blis_info:
#   NOT AVAILABLE
# 创建一个长度为10的空向量
import numpy as np

np.empty(10)
# 找到任何一个数组的内存大小
import numpy as np

data = np.random.randn(2, 2)
print("%d bytes" % (data.size * data.itemsize))
# 输出：

# 32 bytes
# 1
# 从命令行得到numpy中add函数的说明文档
import numpy as np

np.info(np.add)
# 创建一个长度为10并且除了第5个值为1的空向量
import numpy as np

data = np.zeros(10)
data[4] = 5
print(data)
# 创建一个值域范围从10到49的向量
import numpy as np

data = np.arange(10, 50)
print(data)
# 反转一个向量(第一个元素变为最后一个)
import numpy as np

data = np.arange(1, 11)
data = data[::-1]
print(data)
# 创建一个 3x3 并且值从0到8的矩阵
import numpy as np

data = np.arange(9).reshape(3, 3)
print(data)
# 找到数组[1,2,0,0,4,0]中非0元素的位置索引
import numpy as np

data = np.array([1, 2, 0, 0, 4, 0])
nz = np.nonzero(data)
print(nz)
import numpy as np

data = np.array([1, 2, 0, 0, 4, 0])
for x in np.arange(0, len(data)):
    if data[x] != 0:
        print(x)
# 创建一个 3 × 3 3 \times 33×3 的单位矩阵
import numpy as np

data = np.eye(3)
# 创建一个3 × 3 × 3 3 \times 3 \times 33×3×3的随机数组
import numpy as np

data = np.random.random((3, 3, 3))
print(data)
# 创建一个 10 × 10 10 \times 1010×10 的随机数组并找到它的最大值和最小值
import numpy as np

data = np.random.random((10, 10))
print(np.max(data))
print(np.min(data))
# 创建一个长度为30的随机向量并找到它的平均值
import numpy as np

data = np.random.random(10)
print(np.mean(data))
# 创建一个二维数组，其中边界值为1，其余值为0
import numpy as np

data = np.zeros((3, 3))
data[0, :] = 1  # 第1行
data[-1, :] = 1  # 最后1行
data[:, 0] = 1  # 第1列
data[:, -1] = 1  # 最后1列
print(data)
# 对于一个已存在数组，添加一个用0填充的边界
import numpy as np

data = np.ones((5, 5))
data = np.pad(data, pad_width=1, mode='constant', constant_values=0)

# 以下表达式运行的结果分别是什么?
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
# 输出：

# NaN = not a number, inf = infinity
# nan
# False
# False
# nan
# False
# 创建一个 5x5的矩阵，并设置值1,2,3,4落在其对角线下方位置
import numpy as np

data = np.diag(1 + np.arange(4), k=-1)
print(data)
# 创建一个8x8 的矩阵，并且设置成棋盘样式
import numpy as np

data = np.zeros((8, 8), dtype=int)
data[1::2, ::2] = 1
data[::2, 1::2] = 1
print(data)
# 一个 (6,7,8) 形状的数组，其第100个元素的索引(x,y,z)是什么
import numpy as np

print(np.unravel_index(100, (6, 7, 8)))
# 用tile函数去创建一个 8 × 8 8 \times 88×8的棋盘样式矩阵
import numpy as np

data = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(data)
# 对一个5 × 5 5 \times 55×5的随机矩阵做归一化
import numpy as np

data = np.random.random((5, 5))
data_max, data_min = data.max(), data.min();
data = (data - data_min) / (data_max - data_min);
print(data)

# 创建一个将颜色描述为(RGBA)四个无符号字节的自定义dtype
import numpy as np

color = np.dtype([("r", np.ubyte, (1,)),
                  ("g", np.ubyte, (1,)),
                  ("b", np.ubyte, (1,)),
                  ("a", np.ubyte, (1,))])
print(color)
# 一个5 × 3 5 \times 35×3的矩阵与一个3 × 2 3 \times 23×2的矩阵相乘，实矩阵乘积是什么？
import numpy as np

data_1 = np.random.randn(5, 3)
data_2 = np.random.randn(3, 2)
data = np.dot(data_1, data_2)
print(data)
# 给定一个一维数组，对其在3到8之间的所有元素取反
import numpy as np

data = np.arange(11)
data[(data > 3) & (data < 8)] *= -1
print(data)
# 26.下面脚本运行后的结果是什么?

print(sum(range(5), -1))  # 对提供的可迭代对象进行迭代，对值求和，然后加-1
# 1
# 输出：

from numpy import *

print(sum(range(5), -1))  # 将提供的列表所有值求和
# 输出
#
# 10
# 1
# 考虑一个整数向量Z,下列表达合法的是哪个?
import numpy as np

Z = np.arange(1, 6);
print(Z ** Z)
print(2 << Z >> 2)
print(Z < - Z)
print(1j * Z)
print(Z / 1 / 1)
print(Z < Z > Z)
# 下列表达式的结果分别是什么?
import numpy as np

print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
# 如何从零位对浮点数组做舍入?
import numpy as np

# 从均匀[0,1）分布中抽取样本
data = np.random.uniform(-10, +10, 10)
print(np.copysign(np.ceil(np.abs(data)), data))
# 如何找到两个数组中的共同元素?
import numpy as np

data_1 = np.arange(1, 6)
data_2 = np.arange(3, 8)
print(np.intersect1d(data_1, data_2))
# 如何忽略所有的 numpy 警告(尽管不建议这么做)?
import numpy as np

defaults = np.seterr(all="ignore")
data = np.ones(1) / 0
# 下面的表达式是正确的吗?
import numpy as np

print(np.sqrt(-1) == np.emath.sqrt(-1))
# 如何得到昨天，今天，明天的日期?
import numpy as np

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Yesterday is " + str(yesterday))
print("Today is " + str(today))
print("Tomorrow is " + str(tomorrow))
# 如何得到所有与2016年7月对应的日期?
import numpy as np

data = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(data)
# 如何直接在位计算(A+B)*(-A/2)(不建立副本)?
import numpy as np

A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
# 用五种不同的方法去提取一个随机数组的整数部分
import numpy as np

data = np.random.uniform(0, 10, 10)
# 减去小数位
print(data - data % 1)
# 向下取整
print(np.floor(data))
# 向上取整后减1
print(np.ceil(data) - 1)
# 将数据格式变为int
print(data.astype(int))
# 截断函数trunc，丢弃带符号数的小数部分
print(np.trunc(data))
# 创建一个5 × 5 5 \times 55×5的矩阵，其中每行的数值范围从0到4
import numpy as np

data = np.zeros((5, 5))
data += np.arange(0, 5)
print(data)
# 通过考虑一个可生成10个整数的函数，来构建一个数组
import numpy as np


def temp():
    return np.arange(0, 10)


data = temp()
print(data)
# 创建一个长度为10的随机向量，其值域范围从0到1，但是不包括0和1
import numpy as np

# np.linspace()在指定的间隔内返回均匀间隔的数字
# endpoint设置将不包括1
# [1:]将0剔除
data = np.linspace(0, 1, 11, endpoint=False)[1:]
print(data)
# 创建一个长度为10的随机向量，并将其排序
import numpy as np

data = np.random.randn(10)
data.sort()
print(data)
# 对于一个小数组，如何用比 np.sum更快的方式对其求和？
import numpy as np

data = np.arange(10)
print(np.add.reduce(data))

# 对于两个随机数组A和B，检查它们是否相等
import numpy as np

A = np.arange(1, 5)
B = np.arange(3, 7)
# np.allclose比较两个array是不是每一元素都相等
equal = np.allclose(A, B)
print(equal)

# 创建一个只读数组(read-only)
import numpy as np

Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1

# 将笛卡尔坐标下的一个10 × 2 10 \times 210×2的矩阵转换为极坐标形式
import numpy as np

Z = np.random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X ** 2 + Y ** 2)
T = np.arctan2(Y, X)
print(R)
print(T)

# 创建一个长度为10的向量，并将向量中最大值替换为1
import numpy as np

data = np.arange(0, 8)
data_max = data.max()
data[data == data_max] = 1
print(data)
# 或

import numpy as np

data = np.arange(0, 8)
data[data.argmax()] = 0
print(data)
# 创建一个结构化数组，并实现 x xx 和 y yy 坐标覆盖[ 0 , 1 ] ∼ [ 0 , 1 ] [0,1] \sim [0,1][0,1]∼[0,1]
import numpy as np

data = np.zeros((5, 5), [('x', float), ('y', float)])
data['x'], data['y'] = np.meshgrid(np.linspace(0, 1, 5),
                                   np.linspace(0, 1, 5))
print(data)
# 给定两个数组X XX和Y YY，构造Cauchy矩阵C ( C i j = 1 / ( x i − y j ) ) C(C_{ij} =1/(x_i - y_j))C(C
# ij =1/(xi −yj))
import numpy as np

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

# 打印每个numpy标量类型的最小值和最大值
import numpy as np

for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
# 如何打印一个数组中的所有数值?
np.set_printoptions(threshold=np.nan)
data = np.zeros((16, 16))
print(data)
# 给定标量时，如何找到数组中最接近标量的值
data = np.arange(100)
v = np.random.uniform(0, 100)
index = (np.abs(data - v)).argmin()
print(data[index])
