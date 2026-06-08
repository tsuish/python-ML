"""
@File: 第一周-demo2-单元1聚类-聚类之KMeans_31省市居民家庭消费调查.py
@Author: cuishuohao
@Date: 2026/5/29 23:46
"""

import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData,retCityName

# 聚成四类
def clu_4():
    data,cityName = loadData('city.txt')
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    CityCluster = [[],[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f"%expenses[i])
        print(CityCluster[i])

# 聚成三类
def clu_3():
    data,cityName = loadData('city.txt')
    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    CityCluster = [[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f"%expenses[i])
        print(CityCluster[i])

# 聚成两类
def clu_2():
    data,cityName = loadData('city.txt')
    km = KMeans(n_clusters=2)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    CityCluster = [[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f"%expenses[i])
        print(CityCluster[i])

# clu_4()
# clu_3()
clu_2()
