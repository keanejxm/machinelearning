#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  pandas_answer.py
:time  2023/6/19 17:14
:desc  https://www.heywhale.com/mw/project/59e77a636d213335f38daec2
"""
# 步骤1 导入必要的库
# 运行以下代码
import pandas as pd

# 步骤2 从如下地址导入数据集
# 运行以下代码
path1 = "../input/pandas_exercise/pandas_exercise/exercise_data/chipotle.tsv"  # chipotle.tsv
# 步骤3 将数据集存入一个名为chipo的数据框内
# 运行以下代码
chipo = pd.read_csv(path1, sep='\t')
# 步骤4 查看前10行内容
# 运行以下代码
chipo.head(10)
# order_id	quantity	item_name	choice_description	item_price
# 0	1	1	Chips and Fresh Tomato Salsa	NaN	$2.39
# 1	1	1	Izze	[Clementine]	$3.39
# 2	1	1	Nantucket Nectar	[Apple]	$3.39
# 3	1	1	Chips and Tomatillo-Green Chili Salsa	NaN	$2.39
# 4	2	2	Chicken Bowl	[Tomatillo-Red Chili Salsa (Hot), [Black Beans...	$16.98
# 5	3	1	Chicken Bowl	[Fresh Tomato Salsa (Mild), [Rice, Cheese, Sou...	$10.98
# 6	3	1	Side of Chips	NaN	$1.69
# 7	4	1	Steak Burrito	[Tomatillo Red Chili Salsa, [Fajita Vegetables...	$11.75
# 8	4	1	Steak Soft Tacos	[Tomatillo Green Chili Salsa, [Pinto Beans, Ch...	$9.25
# 9	5	1	Steak Burrito	[Fresh Tomato Salsa, [Rice, Black Beans, Pinto...	$9.25
# 步骤6 数据集中有多少个列(columns)
# 运行以下代码
chipo.shape[1]
# 步骤7 打印出全部的列名称
# 运行以下代码
chipo.columns
# Index(['order_id', 'quantity', 'item_name', 'choice_description',
#        'item_price'],
#       dtype='object')
# 步骤8 数据集的索引是怎样的
# 运行以下代码
chipo.index
# RangeIndex(start=0, stop=4622, step=1)
# 步骤9 被下单数最多商品(item)是什么?
# 运行以下代码，做了修正
c = chipo[['item_name', 'quantity']].groupby(['item_name'], as_index=False).agg({'quantity': sum})
c.sort_values(['quantity'], ascending=False, inplace=True)
c.head()
# item_name	quantity
# 17	Chicken Bowl	761
# 18	Chicken Burrito	591
# 25	Chips and Guacamole	506
# 39	Steak Burrito	386
# 10	Canned Soft Drink	351
# 步骤10 在item_name这一列中，一共有多少种商品被下单？
# 运行以下代码
chipo['item_name'].nunique()
# 步骤11 在choice_description中，下单次数最多的商品是什么？
# 运行以下代码，存在一些小问题
chipo['choice_description'].value_counts().head()
# [Diet Coke]                                                                          134
# [Coke]                                                                               123
# [Sprite]                                                                              77
# [Fresh Tomato Salsa, [Rice, Black Beans, Cheese, Sour Cream, Lettuce]]                42
# [Fresh Tomato Salsa, [Rice, Black Beans, Cheese, Sour Cream, Guacamole, Lettuce]]     40
# Name: choice_description, dtype: int64
# 步骤12 一共有多少商品被下单？
# 运行以下代码
total_items_orders = chipo['quantity'].sum()
# total_items_orders
# 4972
# 步骤13 将item_price转换为浮点数
# 运行以下代码
dollarizer = lambda x: float(x[1:-1])
chipo['item_price'] = chipo['item_price'].apply(dollarizer)
# 步骤14 在该数据集对应的时期内，收入(revenue)是多少
# 运行以下代码,已经做更正
chipo['sub_total'] = round(chipo['item_price'] * chipo['quantity'], 2)
chipo['sub_total'].sum()
# 39237.02
# 步骤15 在该数据集对应的时期内，一共有多少订单？
# 运行以下代码
chipo['order_id'].nunique()
# 1834
# 步骤16 每一单(order)对应的平均总价是多少？
# 运行以下代码，已经做过更正
chipo[['order_id', 'sub_total']].groupby(by=['order_id']
                                         ).agg({'sub_total': 'sum'})['sub_total'].mean()
# 21.39423118865867
# 步骤17 一共有多少种不同的商品被售出？
# 运行以下代码
chipo['item_name'].nunique()

# 步骤1 - 导入必要的库
# 运行以下代码
import pandas as pd

# 步骤2 - 从以下地址导入数据集
# 运行以下代码
path2 = "../input/pandas_exercise/exercise_data/Euro2012_stats.csv"  # Euro2012_stats.csv
# 步骤3 - 将数据集命名为euro12
# 运行以下代码
euro12 = pd.read_csv(path2)

# 步骤4只选取Goals这一列
# 运行以下代码
euro12.Goals
# 步骤5有多少球队参与了2012欧洲杯？
# 运行以下代码
euro12.shape[0]
# 步骤6该数据集中一共有多少列(columns)?
# 运行以下代码
euro12.info()
# 步骤7将数据集中的列Team, YellowCards和RedCards单独存为一个名叫discipline的数据框
# 运行以下代码
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
# 步骤8对数据框discipline按照先RedCards再YellowCards进行排序
# 运行以下代码
discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False)
# 步骤9计算每个球队拿到的黄牌数的平均值
# 运行以下代码
round(discipline['Yellow Cards'].mean())

# 步骤10找到进球数Goals超过6的球队数据
# 运行以下代码
euro12[euro12.Goals > 6]
# 步骤11选取以字母G开头的球队数据
# 运行以下代码
euro12[euro12.Team.str.startswith('G')]
# 步骤12选取前7列
# 运行以下代码
euro12.iloc[:, 0:7]
# 步骤13选取除了最后3列之外的全部列
# 运行以下代码
euro12.iloc[:, :-3]

# 步骤14找到英格兰(England)、意大利(Italy)和俄罗斯(Russia)的射正率(ShootingAccuracy)
# 运行以下代码
euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]

# 练习3 - 数据分组
# 探索酒类消费数据
# 步骤1导入必要的库
# 运行以下代码
import pandas as pd

# 步骤2从以下地址导入数据
# 运行以下代码
path3 = '../input/pandas_exercise/pandas_exercise/exercise_data/drinks.csv'  # 'drinks.csv'
# 步骤3将数据框命名为drinks
# 运行以下代码
drinks = pd.read_csv(path3)
drinks.head()
# 步骤4哪个大陆(continent)平均消耗的啤酒(beer)更多？
# 运行以下代码
drinks.groupby('continent').beer_servings.mean()
# 步骤5打印出每个大陆(continent)的红酒消耗(wine_servings)的描述性统计值
# 运行以下代码
drinks.groupby('continent').wine_servings.describe()
# 步骤6打印出每个大陆每种酒类别的消耗平均值
# 运行以下代码
drinks.groupby('continent').mean()
# 步骤7打印出每个大陆每种酒类别的消耗中位数
# 运行以下代码
drinks.groupby('continent').median()
# 步骤8打印出每个大陆对spirit饮品消耗的平均值，最大值和最小值
# 运行以下代码
drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
# 练习4 - Apply函数探索1960 - 2014美国犯罪数据

# 步骤1导入必要的库
# 运行以下代码
import numpy as np
import pandas as pd

# 步骤2从以下地址导入数据集
# 运行以下代码
path4 = '../input/pandas_exercise/pandas_exercise/exercise_data/US_Crime_Rates_1960_2014.csv'  # "US_Crime_Rates_1960_2014.csv"
# 步骤3将数据框命名为crime
# 运行以下代码
crime = pd.read_csv(path4)
crime.head()
# 步骤4每一列(column)的数据类型是什么样的？
# 运行以下代码
crime.info()
# 注意到了吗，Year的数据类型为int64，但是pandas有一个不同的数据类型去处理时间序列(timeseries)，我们现在来看看。

# 步骤5将Year的数据类型转换为datetime64
# 运行以下代码
crime.Year = pd.to_datetime(crime.Year, format='%Y')
crime.info()
# 步骤6将列Year设置为数据框的索引
# 运行以下代码
crime = crime.set_index('Year', drop=True)
crime.head()
# 步骤7删除名为Total的列
# 运行以下代码
del crime['Total']
crime.head()
# 步骤8按照Year对数据框进行分组并求和*注意Population这一列，若直接对其求和，是不正确的 **

# 更多关于 .resample 的介绍
# (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)
# 更多关于 Offset Aliases的介绍
# (http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
# 运行以下代码
crimes = crime.resample('10AS').sum()  # resample a time series per decades

# 用resample去得到“Population”列的最大值
population = crime['Population'].resample('10AS').max()

# 更新 "Population"
crimes['Population'] = population

# 步骤9何时是美国历史上生存最危险的年代？
# 运行以下代码
crime.idxmax(0)
# 练习5 - 合并探索虚拟姓名数据

# 步骤1导入必要的库
# 运行以下代码
import numpy as np
import pandas as pd

# 步骤2按照如下的元数据内容创建数据框
# 运行以下代码
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
# 步骤3将上述的数据框分别命名为data1, data2, data3
# 运行以下代码
data1 = pd.DataFrame(raw_data_1, columns=['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns=['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns=['subject_id', 'test_id'])
# 步骤4将data1和data2两个数据框按照行的维度进行合并，命名为all_data
# 运行以下代码
all_data = pd.concat([data1, data2])
# 步骤5将data1和data2两个数据框按照列的维度进行合并，命名为all_data_col
# 运行以下代码
all_data_col = pd.concat([data1, data2], axis=1)
# 步骤6打印data3
# 运行以下代码
data3
# 步骤7按照subject_id的值对all_data和data3作合并
# 运行以下代码
pd.merge(all_data, data3, on='subject_id')
# 步骤8对data1和data2按照subject_id作连接
# 运行以下代码
pd.merge(data1, data2, on='subject_id', how='inner')
# 步骤9找到data1和data2合并之后的所有匹配结果
# 运行以下代码
pd.merge(data1, data2, on='subject_id', how='outer')
# 练习6 - 统计探索风速数据
#
# 步骤1导入必要的库
# 运行以下代码
import pandas as pd
import datetime

# 步骤2从以下地址导入数据
import pandas as pd

# 运行以下代码
path6 = "../input/pandas_exercise/pandas_exercise/exercise_data/wind.data"  # wind.data
# 步骤3将数据作存储并且设置前三列为合适的索引
import datetime

# 运行以下代码
data = pd.read_table(path6, sep="\s+", parse_dates=[[0, 1, 2]])
data.head()


# 步骤4 2061年？我们真的有这一年的数据？创建一个函数并用它去修复这个bug


# 运行以下代码
def fix_century(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return datetime.date(year, x.month, x.day)


# apply the function fix_century on the column and replace the values to the right ones
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_century)

# data.info()
data.head()
# 步骤5将日期设为索引，注意数据类型，应该是datetime64[ns]
# 运行以下代码
# transform Yr_Mo_Dy it to date type datetime64
data["Yr_Mo_Dy"] = pd.to_datetime(data["Yr_Mo_Dy"])

# set 'Yr_Mo_Dy' as the index
data = data.set_index('Yr_Mo_Dy')

data.head()
# data.info()
# 步骤6对应每一个location，一共有多少数据值缺失
# 运行以下代码
data.isnull().sum()
# 步骤7对应每一个location，一共有多少完整的数据值
# 运行以下代码
data.shape[0] - data.isnull().sum()
# 步骤8对于全体数据，计算风速的平均值
# 运行以下代码
data.mean().mean()
# 步骤9 创建一个名为loc_stats的数据框去计算并存储每个location的风速最小值，最大值，平均值和标准差
# 运行以下代码
loc_stats = pd.DataFrame()

loc_stats['min'] = data.min()  # min
loc_stats['max'] = data.max()  # max
loc_stats['mean'] = data.mean()  # mean
loc_stats['std'] = data.std()  # standard deviations
# 步骤10 创建一个名为day_stats的数据框去计算并存储所有location的风速最小值，最大值，平均值和标准差
# 运行以下代码
# create the dataframe
day_stats = pd.DataFrame()

# this time we determine axis equals to one so it gets each row.
day_stats['min'] = data.min(axis=1)  # min
day_stats['max'] = data.max(axis=1)  # max
day_stats['mean'] = data.mean(axis=1)  # mean
day_stats['std'] = data.std(axis=1)  # standard deviations

day_stats.head()
# 步骤11 对于每一个location，计算一月份的平均风速注意，1961年的1月和1962年的1月应该区别对待

# 运行以下代码
# creates a new column 'date' and gets the values from the index
data['date'] = data.index

# creates a column for each value from date
data['month'] = data['date'].apply(lambda date: date.month)
data['year'] = data['date'].apply(lambda date: date.year)
data['day'] = data['date'].apply(lambda date: date.day)

# gets all value from the month 1 and assign to janyary_winds
january_winds = data.query('month == 1')

# gets the mean from january_winds, using .loc to not print the mean of month, year and day
january_winds.loc[:, 'RPT':"MAL"].mean()
# 步骤12 对于数据记录按照年为频率取样
# 运行以下代码
data.query('month == 1 and day == 1')
# 步骤13 对于数据记录按照月为频率取样
# 运行以下代码
data.query('day == 1')

# 练习7 - 可视化探索泰坦尼克灾难数据

# 步骤1 导入必要的库
# 运行以下代码
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 步骤2 从以下地址导入数据
# 运行以下代码
path7 = '../input/pandas_exercise/pandas_exercise/exercise_data/train.csv'  # train.csv
# 步骤3 将数据框命名为titanic
# 运行以下代码
titanic = pd.read_csv(path7)
titanic.head()
# 步骤4 将PassengerId设置为索引
# 运行以下代码
titanic.set_index('PassengerId').head()
# 步骤5 绘制一个展示男女乘客比例的扇形图
# 运行以下代码
# sum the instances of males and females
males = (titanic['Sex'] == 'male').sum()
females = (titanic['Sex'] == 'female').sum()

# put them into a list called proportions
proportions = [males, females]

# Create a pie chart
plt.pie(
    # using proportions
    proportions,

    # with the labels being officer names
    labels=['Males', 'Females'],

    # with no shadows
    shadow=False,

    # with colors
    colors=['blue', 'red'],

    # with one slide exploded out
    explode=(0.15, 0),

    # with the start angle at 90%
    startangle=90,

    # with the percent listed as a fraction
    autopct='%1.1f%%'
)

# View the plot drop above
plt.axis('equal')

# Set labels
plt.title("Sex Proportion")

# View the plot
plt.tight_layout()
plt.show()

# 步骤6 绘制一个展示船票Fare, 与乘客年龄和性别的散点图
# 运行以下代码
# creates the plot using
lm = sns.lmplot(x='Age', y='Fare', data=titanic, hue='Sex', fit_reg=False)

# set title
lm.set(title='Fare x Age')

# get the axes object and tweak it
axes = lm.axes
axes[0, 0].set_ylim(-5, )
axes[0, 0].set_xlim(-5, 85)

# 步骤7 有多少人生还？
# 运行以下代码
titanic.Survived.sum()
# 步骤8 绘制一个展示船票价格的直方图
# 运行以下代码
# sort the values from the top to the least value and slice the first 5 items
df = titanic.Fare.sort_values(ascending=False)
df

# create bins interval using numpy
binsVal = np.arange(0, 600, 10)
binsVal

# create the plot
plt.hist(df, bins=binsVal)

# Set the title and labels
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Payed Histrogram')

# show the plot
plt.show()

# 练习8 - 创建数据框探索Pokemon数据
# 步骤1 导入必要的库
# 运行以下代码
import pandas as pd

# 步骤2 创建一个数据字典
# 运行以下代码
raw_data = {"name": ['Bulbasaur', 'Charmander', 'Squirtle', 'Caterpie'],
            "evolution": ['Ivysaur', 'Charmeleon', 'Wartortle', 'Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no', 'yes', 'no']
            }
# 步骤3 将数据字典存为一个名叫pokemon的数据框中
# 运行以下代码
pokemon = pd.DataFrame(raw_data)
pokemon.head()
# 步骤4 数据框的列排序是字母顺序，请重新修改为name, type, hp, evolution, pokedex这个顺序
# 运行以下代码
pokemon = pokemon[['name', 'type', 'hp', 'evolution', 'pokedex']]
pokemon
# 步骤5 添加一个列place
# 运行以下代码
pokemon['place'] = ['park', 'street', 'lake', 'forest']
# 步骤6 查看每个列的数据类型
# 运行以下代码
pokemon.dtypes
# 练习9 - 时间序列探索Apple公司股价数据
# 步骤1 导入必要的库
# 运行以下代码
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt

# 步骤2 数据集地址
# 运行以下代码
path9 = '../input/pandas_exercise/pandas_exercise/exercise_data/Apple_stock.csv'  # Apple_stock.csv
# 步骤3 读取数据并存为一个名叫apple的数据框
# 运行以下代码
apple = pd.read_csv(path9)
apple.head()
# 步骤4 查看每一列的数据类型
# 运行以下代码
apple.dtypes
# 步骤5 将Date这个列转换为datetime类型
# 运行以下代码
apple.Date = pd.to_datetime(apple.Date)
apple['Date'].head()
# 步骤6 将Date设置为索引
# 运行以下代码
apple = apple.set_index('Date')
apple.head()
# 步骤7 有重复的日期吗？
# 运行以下代码
apple.index.is_unique
True
# 步骤8 将index设置为升序
# 运行以下代码
apple.sort_index(ascending=True).head()
# 步骤9 找到每个月的最后一个交易日(business day)
# 运行以下代码
apple_month = apple.resample('BM')
apple_month.head()
# 步骤10 数据集中最早的日期和最晚的日期相差多少天？
# 运行以下代码
(apple.index.max() - apple.index.min()).days

# 步骤11 在数据中一共有多少个月？
# 运行以下代码
apple_months = apple.resample('BM').mean()
len(apple_months.index)
404
# 步骤12 按照时间顺序可视化AdjClose值
# 运行以下代码
# makes the plot and assign it to a variable
appl_open = apple['Adj Close'].plot(title="Apple Stock")

# changes the size of the graph
fig = appl_open.get_figure()
fig.set_size_inches(13.5, 9)

# 练习10 - 删除数据探索Iris纸鸢花数据
# 步骤1 导入必要的库
# 运行以下代码
import pandas as pd

# 步骤2 数据集地址
# 运行以下代码
path10 = '../input/pandas_exercise/pandas_exercise/exercise_data/iris.csv'  # iris.csv
# 步骤3 将数据集存成变量iris
# 运行以下代码
iris = pd.read_csv(path10)
iris.head()
# 步骤4 创建数据框的列名称
iris = pd.read_csv(path10, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
iris.head()
# 步骤5 数据框中有缺失值吗？
# 运行以下代码
pd.isnull(iris).sum()

# 步骤6 将列petal_length的第10到19行设置为缺失值
# 运行以下代码
iris.iloc[10:20, 2:3] = np.nan
iris.head(20)
# 步骤7 将缺失值全部替换为1.0
# 运行以下代码
iris.petal_length.fillna(1, inplace=True)
iris

# 步骤8 删除列class
# 运行以下代码
del iris['class']
iris.head()
# 步骤9 将数据框前三行设置为缺失值
# 运行以下代码
iris.iloc[0:3, :] = np.nan
iris.head()
# 步骤10 删除有缺失值的行
# 运行以下代码
iris = iris.dropna(how='any')
iris.head()
# 步骤11 重新设置索引
# 运行以下代码
iris = iris.reset_index(drop=True)
iris.head()
