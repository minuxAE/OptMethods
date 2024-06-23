import numpy as np
import copy

"""
蝴蝶算法
"""

def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j]=(ub[j] - lb[j]) * np.random.random() + lb[j]
    return X

def fun(x):
    fitness = np.sum(x**2)
    return fitness

def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

def Fitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def BOA(pop, dim, lb, ub, maxIter, fun):
    p = 0.8 # 全局搜索和局部搜索的切换概率
    