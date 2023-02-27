#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  practice_8.py
:time  2023/2/2 11:02
:desc  张量的结构操作
"""
import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3], dtype=tf.float32)
tf.print(a)

b = tf.range(1, 10, delta=2)  # 以start为起点，delta为变化量，生成不超过，且不包含limit的Tensor
tf.print(b)

c = tf.linspace(0.0, 2 * 3.14,
                100)  # 这个函数主要的参数就这三个，start代表起始的值，end表示结束的值，num表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。start和end这两个数字必须是浮点数，不能是整数，如果是整数会出错的
tf.print(c)

d = tf.zeros([3, 3])
tf.print(d)

e = tf.ones([3, 3])
f = tf.zeros_like(e, dtype=tf.float32)  # 操作返回与所有元素设置为零的tensor具有相同类型和形状的张量.或者,您可以使用dtype指定返回张量的新类型.
tf.print(e)
tf.print(f)

g = tf.fill([3, 2], 5)  # 创建一个m*n矩阵并用5填充
tf.print(g)

# 均匀分布随机
tf.random.set_seed(1.0)  # 设置全局随机种子
h = tf.random.uniform([5], minval=0, maxval=10)  # 用于从均匀分布中输出随机值
tf.print(h)

# 正态分布随机
i = tf.random.normal([3, 3], mean=0.0, stddev=1.0)  # 服从指定正态分布的序列”中随机取出指定个数的值
tf.print(i)

# 正态分布随机，剔除2倍方差以外数据重新生成
j = tf.random.truncated_normal([5, 5], mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(j)

k = tf.eye(3, 3)  # 单位矩阵
tf.print(k)
l = tf.linalg.diag([1, 2, 3])  # 对角阵
tf.print(l)

# 索引切片

tf.random.set_seed(3)
m = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)
tf.print(m)

# 取第0行
tf.print(m[0])
# 倒数第一行
tf.print(m[-1])
# 第一行第三列
tf.print(m[1][3])
tf.print(m[1, 3])

# 第一行至第三行
tf.print(m[1:4, :])
tf.print(tf.slice(m, [1, 0], [3, 5]))

# 第一行至最后一行，第0列到最后一列，每隔两列取一列
tf.print(m[1:4, :4:2])

# 对于变量来说，还可以使用索引和切片修改部分元素
n = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
n[1, :].assign(tf.constant([0.0, 0.0]))
tf.print(n)

# 省略号可以表示多个冒号
o = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
tf.print(o)
tf.print(o[..., 1])

# 虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4×10×7的张量来表示。
scores = tf.random.uniform([4, 10, 7], minval=0, maxval=100, dtype=tf.int32)
tf.print(scores)
# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.gather(scores, [0, 5, 9], axis=1)
tf.print(p)
# 抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = tf.gather(tf.gather(scores, [0, 5, 9], axis=1), [1, 3, 6], axis=2)
tf.print(q)

# 抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
# indices的长度为采样样本的个数，每个元素为采样位置的坐标
r = tf.gather_nd(scores, indices=[[0, 0], [2, 4], [3, 6]])
tf.print(r)

# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
s = tf.boolean_mask(scores, [True, False, False, False, False, True, False, False, False, True], axis=1)
tf.print(s)

# 抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
t = tf.boolean_mask(scores,
                    [
                        [True, False, False, False, False, False, False, False, False, False],
                        [False, False, False, False, True, False, False, False, False, False],
                        [False, False, False, False, False, False, True, False, False, False],
                        [False, False, False, False, False, False, False, False, False, False]
                    ])
tf.print(t)

# 利用tf.boolean_mask可以实现布尔索引
# 找到矩阵中小于0的元素
u = tf.constant([[-1, 1, -1], [2, 3, -2], [3, -3, 3]], dtype=tf.float32)
tf.print(u, "\n")
tf.print(tf.boolean_mask(u, u < 0), "\n")
tf.print(u[u < 0])

# 找到张量中小于0的元素,将其换成np.nan得到新的张量
# tf.where和np.where作用类似，可以理解为if的张量版本
v = tf.where(u < 0, tf.fill(u.shape, np.nan), u)
tf.print(v)

# 如果where只有一个参数，将返回所有满足条件的位置坐标
indices = tf.where(u < 0)
tf.print(indices)

# 将张量的第[0,0]和[2,1]两个位置元素替换为0得到新的张量
w = u - tf.scatter_nd([[0, 0], [2, 1]], [u[0, 0], u[2, 1]], u.shape)
tf.print(w)

# scatter_nd的作用和gather_nd有些相反
# 可以将某些值插入到一个给定shape的全0的张量的指定位置处。
indices = tf.where(c < 0)
tf.scatter_nd(indices, tf.gather_nd(c, indices), c.shape)

# 维度变换
# tf.reshape:可以改变张量的形状
# tf.squeeze:可以减少维度
# tf.expand_dims:可以增加维度
# tf.transpose：可以交换维度


# tf.reshape可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序
x = tf.random.uniform(shape=[1, 3, 3, 2], minval=0, maxval=255, dtype=tf.int32)
tf.print(x.shape)
tf.print(x)

# 改成（3,6）的形状
y = tf.reshape(x, [3, 6])
tf.print(y.shape)
tf.print(y)
# 改回（1,3,3,2）的形状
z = tf.reshape(y, [1, 3, 3, 2])
tf.print(z.shape)
tf.print(z)

# 如果张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度

aa = tf.squeeze(x)
tf.print(aa.shape)
tf.print(aa)

# 在第0维插入长度为1的一个维度
ab = tf.expand_dims(aa, axis=0)
tf.print(ab.shape)
tf.print(ab)

# tf.transpose可以交换张量维度，与tf.reshape不同他会改变张量元素的存储顺序，常用语图片存储格式的变换上
ac = tf.random.uniform([100, 600, 600, 4], minval=0, maxval=255, dtype=tf.int32)
tf.print(ac.shape)
ad = tf.transpose(ac, perm=[1, 3, 2, 0])
tf.print(ad.shape)

# 合并分割
ae = tf.constant([[1.0, 2.0], [3.0, 4.0]])
af = tf.constant([[5.0, 6.0], [7.0, 8.0]])
ag = tf.constant([[9.0, 10.0], [11.0, 12.0]])
ah = tf.concat([ae, af, ag], axis=0)
tf.print(ah.shape)
tf.print(ah)

aj = tf.concat([ae, af, ag], axis=1)
tf.print(aj.shape)
tf.print(aj)

# tf.concat和tf.stack有略微的区别，tf.concat是连接，不会增加维度，而tf.stack是堆叠，会增加维度。
ak = tf.stack([ae, af, ag], axis=0)
tf.print(ak.shape)
tf.print(ak)

al = tf.stack([ae, af, ag], axis=1)
tf.print(al.shape)
tf.print(al)

am,an,ao = tf.split(ah,3,axis = 0)
tf.print(am.shape)
tf.print(am)

tf.print(an.shape)
tf.print(an)

tf.print(ao.shape)
tf.print(ao)

ap,aq,ar = tf.split(ah,[2,2,2],axis = 0)
tf.print(ap.shape)
tf.print(ap)
