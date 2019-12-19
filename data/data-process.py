# coding=utf-8

r"""
多标记学习转化为seq2seq任务中涉及的一些数据处理任务，要求组织成数据处理大类，里面包含各个处理函数
"""

# 按照文件名读取data文件下相应.csv数据文件，分离为示例特征矩阵、示例标签矩阵

# 对于标签集，按照标签出现的频率，降序排列标签集

# 分离特征数据集、标签数据集为训练集（特征、标签）、验证集（特征、标签）、推理（特征、标签）

# 利用KDTree算法或BallTree算法存储训练-特征集

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path='C:\\Users\\DongPu\\Desktop\\多标记数据集\\emotions\\emotions.arff.csv'
data=pd.read_csv(path)
dataset=np.array(data)

features = dataset[:, :72]
rlabels = dataset[:, 72:]

counts = []
for i in range(len(rlabels[0])):
    colunm = rlabels[:, i:i + 1]
    count = np.sum(colunm == 1)
    counts.append(count)
index = np.argsort(counts)
idex = index[::-1]
print(idex)
labels = rlabels[:, idex]


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
train = np.column_stack((X_train, y_train))
test = np.column_stack((X_test, y_test))
