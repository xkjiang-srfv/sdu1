# 聚类算法

import random
import pandas as pd
import numpy as np
import copy
import math


# 计算距离
def Dis(dataSet, centroids, k):
    # 处理质心
    # 如果之前分类的个数不够k类
    if len(centroids) < k:
        centroids = np.append(centroids, random.sample(list(dataSet), k-len(centroids)), axis=0)
    
    # 处理节点
    clalist=[]
    for data in dataSet:
        #(np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        diff = np.tile(data, (k, 1)) 
        mul_Diff = np.multiply(diff, centroids)
        mul_Dist = np.sum(mul_Diff, axis=1)   #和  (axis=1表示行)
        clalist.append(mul_Dist) 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist 


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = Dis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmax(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 对新质心，也分配成1-value_sum的形式，否则会出现小数
    for centro in newCentroids:
        # centro是一个一维向量
        sorted_data=np.argsort(centro)  # 排序信息
        value = 1
        for valueIndex in sorted_data:
            centro[valueIndex] = value
            value += 1
    
    # 计算变化量
    # 有可能新分类个数不够k
    if len(newCentroids) != len(centroids):
        changed = 1  # 肯定有变化
    else:
        changed = newCentroids - centroids # 有可能没变化 

    return changed, newCentroids


#确定初始中心点
def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += a*b
    return distance
    

def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers


# 使用k-means分类
def kmeans(dataSet, k):
    # 将dataSet预处理成为算距离需要使用的重要程度矩阵
    valueSet = np.zeros(dataSet.shape, dtype=int)  # 初始矩阵
    for index in range(len(dataSet)):
        data = dataSet[index]
        value = valueSet[index]
        sorted_data=list(map(abs,data))  # 绝对值
        sorted_data=np.argsort(sorted_data)  # 排序信息
        i = 1  # 对于越小的值，分配的i越小
        for valueIndex in sorted_data:
            value[valueIndex] = i
            i += 1

    # 随机取质心
    # centroids = random.sample(dataSet, k)
    centroids=kpp_centers(valueSet, k)
    
    # 更新质心 直到变化量全为0
    i=100
    changed, newCentroids = classify(valueSet, centroids, k)
    # while(i): #while np.any(changed != 0)
    while np.any(changed != 0) and i > 0:
        changed, newCentroids = classify(valueSet, newCentroids, k)
        i=i-1
        print("第{}次迭代".format(100-i))
 
    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
    clalist = Dis(valueSet, centroids, k) 
    minDistIndices = np.argmax(clalist, axis=1)  
    return minDistIndices


def getCluster(input, clusters_num):
    # 对卷积层聚类为4维，对全连接层聚类为2维
    if len(input.shape) == 2:  # 如果是全连接层
        fcValues = input.detach().cpu().numpy()  # 转成numpy
        # input.shape[1]是聚类基本单位的数据个数
        clusterIndex = kmeans(fcValues, clusters_num)  # 分类
    elif len(input.shape) == 4:  # 卷积层
        kernel_size = input.shape[3]  # 卷积核尺寸
        preShape = input.shape[:2]  # 四维数据的前两维
        inputCut = input.view(preShape[0]*preShape[1], kernel_size*kernel_size)  # 降维后的数据，四维到二维
        convValues = inputCut.detach().cpu().numpy()  # 转成numpy
        clusterIndex = kmeans(convValues, clusters_num)  # 分类
        clusterIndex.resize(preShape)
    else:
        clusterIndex = None
    
    return clusterIndex