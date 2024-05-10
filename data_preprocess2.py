# %%
import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np

# 读取summarize.csv
df = pd.read_csv(r'datasets\datasets_data\collected\processed\summarize.csv')

# %%
path = r"datasets\\datasets_data\\collected\\"


# %%
# 将'date'列转换为datetime类型,并提取月份
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')

# 检查数据质量
print("数据形状:", df.shape)
print("数据类型:\n", df.dtypes)
print("缺失值检查:\n", df.isnull().sum())

# 缺失值处理
# 删除'date'列中有缺失值的行
df.dropna(subset=['date'], inplace=True)

# '城乡居民比例'等比例类指标用前后均值填充
ratio_cols = ['城乡居民比例']
# df[ratio_cols] = df[ratio_cols].fillna(method='ffill').fillna(method='bfill')
df[ratio_cols] = df[ratio_cols].ffill().bfill()

# 人口、经济相关指标用月份时间戳进行插值
pop_eco_cols = ['乡村人口数:年', '城镇人口数:年', '农村家庭平均每人年消费性支出:年?', 
                '城镇居民人均可支配收入:财产净收入:年?', '城镇居民人均消费性支出:年?', 
                '农村居民人均可支配收入_全国_年?', '能源生产总量:年?']
for col in pop_eco_cols:
    df[col] = df.groupby('month')[col].apply(lambda x: x.interpolate(method='index'))

# 其他列用均值填充
other_cols = df.columns.difference(['date', 'month'] + ratio_cols + pop_eco_cols)
df[other_cols] = df[other_cols].fillna(df[other_cols].mean())

# 异常值检查与处理
print("异常值检查-五数概括:\n", df.describe())
# 假设'人均能源生产量:年?'列的异常值为大于1000的值
df.loc[df['人均能源生产量:年?'] > 1000, '人均能源生产量:年?'] = 1000  

# 根据variable_choosel中的变量,从summarize.csv中提取相关指标
# 个人层面变量
df['个人收入'] = df['城镇居民人均可支配收入:财产净收入:年?']  # 直接取'城镇居民人均可支配收入:财产净收入:年?'列
df['个人健康'] = 100 - df['城市呼吸系统疾病粗死亡率_当期值_年?']  # 用100减去'城市呼吸系统疾病粗死亡率_当期值_年?',得到个人健康指标

# 微观层面变量
df['环境污染'] = df['大气污染事故次数:年']  # 直接取'大气污染事故次数:年'列
df['能源利用'] = df['能源生产总量:年?'] / df['一次能源生产量:年?']  # '能源生产总量:年?'除以'一次能源生产量:年?',得到能源利用指标

# 中观层面变量 
df['环保投资'] = df['固定资产投资额(不含农户):电力、热力、燃气及水生产和供应业:能源工业:上海市:年'] + df['固定资产投资额(不含农户):能源工业:上海市:年']  # 对两列求和,得到环保投资指标
df['技术水平'] = df['技术市场成交合同金额:能源生产、分配和合理利用:年?']  # 直接取'技术市场成交合同金额:能源生产、分配和合理利用:年?'列

# 宏观层面变量
df['经济发展'] = df['全国一般公共预算支出决算数_医疗卫生与计划生育支出_公共卫生_疾病预防控制机构_当期值_年?'] + df['全国一般公共预算支出决算数_医疗卫生与计划生育支出_医疗保障_疾病应急救助_当期值_年?']  # 对两列求和,得到经济发展指标
df['人口结构'] = df['城乡居民比例']  # 直接取'城乡居民比例'列

# 只保留所需变量
variables = ['date', 'month', '个人收入','个人健康','环境污染','能源利用','环保投资','技术水平','经济发展','人口结构'] 
df = df[variables]

# 将数据类型转换为float
df[variables[2:]] = df[variables[2:]].astype(float)

# 将'month'列设置为索引
df.set_index('month', inplace=True)

# 数据标准化
df = (df - df.min()) / (df.max() - df.min())

print("预处理后的数据:\n", df.head())
df.to_csv("data_preprocessed.csv")  # 保存预处理后的数据


