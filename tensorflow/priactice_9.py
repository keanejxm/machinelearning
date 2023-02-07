#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  priactice_9.py
:time  2023/2/3 16:54
:desc  张量的数学运算
"""
import tensorflow as tf
import numpy as np

# 加
a = tf.constant([[1.0, 2], [-3, 4.0]])
b = tf.constant([[5.0, 6], [7.0, 8.0]])
c = tf.constant([1.0, 8.0])
d = tf.constant([5.0, 6.0])
e = tf.constant([6.0, 7.0])

tf.print(tf.add_n([c, d, e]))

tf.print(a + b)

# 减
tf.print(a - b)
# 乘
tf.print(a * b)
# 除
tf.print(a / b)
# 乘方
tf.print(a ** 2)
tf.print(a ** 0.5)
# 模除（取余）
tf.print(a % 3)
tf.print(tf.math.mod(a, 3))
# 地板除(取商)
tf.print(a // 3)

# 逻辑运算符
tf.print((a >= 2))
tf.print((a >= 2) & (a <= 3))
tf.print((a >= 2) | (a <= 3))
tf.print((a == 5))
tf.print(tf.equal(a, 5))
# 开根
tf.print(tf.sqrt(a))

# 取最大值最小值

tf.print(tf.maximum(c, d))
tf.print(tf.minimum(c, d))

f = tf.range(1, 10)
tf.print(f)

# 计算tensor指定轴方向上的所有元素的累加和
tf.print(tf.reduce_sum(f))
# 计算tensor指定轴方向上的各个元素的平均值
tf.print(tf.reduce_mean(f))
# 计算tensor指定轴方向上的各个元素的最大值
tf.print(tf.reduce_max(f))
# 计算tensor指定轴方向上的各个元素的最小值
tf.print(tf.reduce_min(f))
# 计算tensor指定轴方向上的各个元素的乘积
tf.print(tf.reduce_prod(f))
# 计算tensor指定轴方向上的各个元素的逻辑和（and运算）
# tf.print(tf.reduce_all(f))
# 计算tensor指定轴方向上的各个元素的逻辑或（or运算）
# tf.print(tf.reduce_any(f))

# 张量指定维度进行reduce
g = tf.reshape(f, (3, 3))

tf.print(g)
tf.print(tf.reduce_sum(g, axis=1, keepdims=True))
tf.print(tf.reduce_sum(g, axis=0, keepdims=True))

# bool类型的reduce
h = tf.constant([True, False, False])
i = tf.constant([False, False, True])
tf.print(tf.reduce_all(h))
tf.print(tf.reduce_any(i))

# 利用tf.foldr实现tf.reduce_sum
j = tf.foldr(lambda c, d: c + d, tf.range(10))
tf.print(j)

# cum扫描累积
k = tf.range(1, 10)
tf.print(tf.math.cumsum(k))
tf.print(tf.math.cumprod(k))

# arg最大最小索引
tf.print(tf.argmax(k))
tf.print(tf.argmin(k))

# tf.math.top_k可以用于对张量排序
l = tf.constant([1, 3, 7, 5, 4, 8])
# value, indices = tf.math.top_k(a, 3, sorted=True)
# tf.print(value)
# tf.print(indices)

# 矩阵运算

# 矩阵乘法
m = tf.constant([[1, 2], [3, 4]])
n = tf.constant([[2, 0], [0, 2]])
tf.print(tf.matmul(m, n))

# 矩阵的转置
tf.print(tf.transpose(m))

# 逆矩阵，必须为tf.float32或者tf.double类型
o = tf.constant([[1.0, 2], [3.0, 4]], dtype=tf.float32)
tf.print(tf.linalg.inv(o))

# 矩阵求迹(对角线的和)
tf.print(tf.linalg.trace(o))

# 矩阵求范数
tf.print(tf.linalg.norm(o))

# 矩阵行列式
tf.print(tf.linalg.det(o))

# 矩阵特征值
tf.print(tf.linalg.eigvalsh(o))

# 矩阵求分解
# qr分解
q, r = tf.linalg.qr(o)
tf.print(q)
tf.print(r)
tf.print(q @ r)
# svd分解
v, s, d = tf.linalg.svd(o)
tf.matmul(tf.matmul(s, tf.linalg.diag(v)), d)

# 张量的广播机制

p = tf.constant([1, 2, 3])
q = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
tf.print(p+q) # 等价于b + tf.broadcast_to(q,p.shape)

tf.print(tf.broadcast_to(q,p.shape))


# 计算广播后结果的形状，静态形状，TensorShape类型参数
tf.print(tf.broadcast_static_shape(p.shape,q.shape))


# 计算广播后结果的形状，动态形状 Tensor类型参数
r = tf.constant([1,2,3])
s = tf.constant([[1],[2],[3]])
tf.print(tf.broadcast_dynamic_shape(tf.shape(r),tf.shape(s)))

#广播效果
# r+s #等价于 tf.broadcast_to(r,[3,3]) + tf.broadcast_to(s,[3,3])