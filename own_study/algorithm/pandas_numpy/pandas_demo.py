#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pandas_demo.py
:time  2023/6/19 16:42
:desc  
"""
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}


raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']
            }

# 开始了解数据
    # 导入数据集
    # 将数据集存入一个名为chipo的数据框内
    # 查看前10行内容
    # 查看数据集中有多少列
    # 数据集的索引是怎样的
    # 被下单最多的商品（item）是什么
    # 在（item_name）这一列中，一共有多少种商品被下单
    # 在choice_description下单次数最多的商品是什么
    # 一共有多少商品被下单
    # 将item_price转为浮点数
    # 在该数据集对应的时期内，收入（revenue）是多少
    # 在该数据集对应的时期内，一共有多少订单
    # 每一单（order）对应的平均总价是多少
    # 一共有多少种不同的商品被出售
# 数据过滤与排序
    # 将数据集命名为euro12
    # 只选取Goals这一列
    # 有多少支球队参加了2012欧洲杯
    # 改数据集中一共有多少列
    # 将数据集中的Team,Yellow Cards 和Red Cards单独存为一个名叫discipline的数据框
    # 对数据框discipline按照先Red Card 再Yellow Cards进行排序
    # 计算每个球队难道黄牌数的平均值
    # 找到进球数Goals超过6的球队数据
    # 选取以字母G开头的球队数据
    # 选取前7列
    # 选取除了最后3列之外的全部列
    # 找到英格兰[England]、意大利（italy）和俄罗斯（Russia）的射正率（Shooting Accuracy）
# 数据分组
    # 将数据框命名为drinks
    # 那个大陆（continent）平均消耗的啤酒（beer）更多
    # 打印出每个大陆（continent）的红酒消耗（wine_servings）的描述性统计值
    # 打印出每个大陆每种酒类别的消耗平均值
    # 打印出每个大陆每种酒类别的消耗中位数
    # 打印出每个大陆对spirit饮品消耗的平均值，最大值和最小值
# apply函数
    # 将数据框命名为crime
    # 每一列（column）的数据类型是什么样的
    # 将Year的数据类型转换为datetime64
    # 将列Year设置为数据框索引
    # 删除名为Total的列
    # 按照Year对数据框进行分组并求和
    # 何时是美国历史上生存最危险的年代
# 合并
    # 将上述的数据框分别命名为data1，data2,data3
    # 将data1和data2两个数据框按照行的维度进行合并，命名为all_data
    # 将data1和data2两个数据框按照列的维度进行合并，命名为all_data_col
    # 打印data3
    # 按照subject_id的值对all_data和data3做合并
    # 对data1和data2按照subject_id作连接
    # 找到data1和data2合并之后的所有匹配结果
# 统计
    # 将数据作存储并且设置前三列为合适的索引
    # 2061年？我们真的有这一年的数据？创建一个函数并用它去修复这个bug
    # 将日期设为索引，注意数据类型，应该是datetime64[ns]
    # 对应每一个location，一共有多个数据值缺失
    # 对应每一个location，一共有多少完成数据值
    # 对于全体数据，计算风速的平均值
    # 创建一个名为loc_stats的数据框去计算并存储每个location的风速最小值，最大值，平均值和标准差
    # 创建一个名为day_stats的数据框去计算并存储每个location的风速最小值，最大值，平均值和标准差
    # 对于每一个location，计算一月份的平均风速
    # 对于数据记录按照年为频率取样
    # 对于数据记录按照月为频率取样
# 可视化
    # 将数据框命名为titanic
    # 将Passengerld设置为索引
    # 绘制一个展示男女乘客比例的扇形图
    # 绘制一个展示船票fare，与乘客年龄和性别的散点图
    # 有多少人生还
    # 绘制一个展示船票价格的直方图
# 创建数据框
    # 创建一个数据字典
raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']
            }
    # 将数据字典存在一个名叫pokemon的数据框中
    # 将数据框的列排序是字母顺序，请重新修改为name,type,hp,evolution,pokedex这个顺序
    # 添加一个列place
    # 查看每个列的数据类型
# 时间序列
    # 读取数据并村委一个名叫apple的数据框
    # 查看每一列的数据类型
    # 将Date这个列转为datetime类型
    # 将Date设置为索引
    # 有重复的日期吗
    # 将index设置为升序
    # 找到每个月的最后一个交易日(businessday)
    # 数据集中最早的日期和最晚的日期相差多少天
    # 在数据中一共有多少个月
    # 按照时间顺序可视化Adj Close值
# 删除数据
    # 将数据集存为变量iris
    # 创建数据框的列名称
    # 数据框中有缺失值吗
    # 将列petal_length的第10到19行设置为缺失值
    # 将缺失值全部替换为1.0
    # 删除列class
    # 将数据框前三行设置为缺失值
    # 删除有缺失值的行
    # 重新设置索引
    #