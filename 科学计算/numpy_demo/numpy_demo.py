#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  numpy_demo.py
:time  2023/11/21 16:31
:desc  
"""
"""
1.内存中的ndarray对象
    元数据
        元数据中存放的是目标数组的描述信息、ndim(维度)、shape(数组类型)、dtype(数据类型)、data
    实际数据
         完整的数组数据
2、ndarray数组的特点
    numpy是同质数组，数组中每个元素的数据类型是相同的,每个元素在内存中占的字节是相同的
    numpy下标是从0开始，到n-1结束
3、ndarray数组的创建
    arange()、ones(),zeros()
4、ndarray数组的属性
    shape、dtype、size、切片、
5、ndarray数组的基本数据类型
    bool(布尔)、有符号整数型（int8,int16,int32,int64）、无符号整数型（uint8,uint16,uint32,uint64）
    浮点型（float16,float32,float32）、复数（complex64、complex128）、字符串（str）、时间类型（datetime64,datetime64[D],datetime64[s]）
    字符串：一个字符占4个字节（每个字符用32位的unicode编码表示）
6、自定义复合类型
7、ndarray数组的维度操作
    不会修改原始数据的维度：
        视图变维：reshape()，reval()
        复制变维：flatten()
    修改原始数据：
        就地变维：a.shape() =  ;a.resize()
8、ndarray数组的索引和切片
    一个索引会降低一个维度，切片不会降低数据的维度
9、ndarray数组的掩码操作
    布尔掩码
    索引掩码
10、ndarray数组的组合与拆分
    vstack,vsplit
    
"""
import numpy as np

# 创建
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr)
print(type(arr))
print(arr * 2)
print(arr * arr)

arr1 = np.arange(1, 10, 1)
print(arr1)

arr2 = np.arange(0, 1, 0.1)
print(arr2)

arr3 = np.zeros(shape=(10,))
print(arr3)

arr4 = np.ones(shape=(10, 2))
print(arr4)

arr5 = np.zeros(shape=(3, 2, 3))
print(arr5)

# 属性
ary = np.arange(1, 9)
print(ary)
print(ary.shape)
ary.shape = (2, 4)
print(ary)
print(ary.shape)

ary = np.arange(1, 9)
print(ary.dtype)

ary = ary.astype("float32")
print(ary.dtype)

ary = np.arange(1, 9)
print(ary.size)

ary.shape = (2, 4)
print(ary[1, 2])

ary = np.array([(12, "bbb", "222")], dtype="int32,3str,3str")
print(ary)
data = [
    ("aa", [50, 50, 50], 18),
    ("bb", [70, 70, 70], 19),
    ("zzz", [80, 80, 80], 20)
]
ary = np.array(data, dtype='3str,3int32,int32')
print(ary["f1"])

ary = np.array(data, dtype={"names": ["name", "score", "age"],
                            "formats": ["3str", "3int32", "int32"]})
print(ary)
print(ary["score"])
