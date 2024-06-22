import numpy as np
import copy
"""
蝗虫优化算法的实现
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

"""
社会相互作用力函数
"""
def s_func(r):
    f = 0.5
    l = 1.5
    return f*np.exp(-r/l)-np.exp(-r)

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

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def s_func(r):
    f = 0.5
    l = 1.5
    return f * np.exp(-r/l)-np.exp(-r)

def GOA(pop, dim, lb, ub, maxIter, fun):
    EPS = 2**-50
    # 定义参数c的范围
    cMax = 1
    cMin = 0.0004
    X = initialization(pop, ub, lb, dim)
    fitness = Fitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex) # 对种群进行排序
    GbestScore = copy.copy(fitness[0])
    GbestPosition = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])
    GrassHopperPositions_temp = np.zeros([pop, dim]) # 临时存放位置

    for t in range(maxIter):
        c = cMax - t*((cMax - cMin) / maxIter)
        print('Iteration: ', t)

        for i in range(pop):
            Temp = X.T
            S_i_total = np.zeros([dim, 1])
            for k in range(0, dim-1, 2):
                S_i = np.zeros([2, 1])
                for j in range(pop):
                    if i != j:
                        Dist = distance(Temp[k:k+2, j], Temp[k:k+2, i]) # 计算两只蝗虫的距离
                        r_ij_vec = (Temp[k:k+2, j] - Temp[k:k+2, i]) / (Dist + EPS) # 距离单位向量
                        xj_xi = 2+Dist%2 # 计算 |xjd - xid|
                        s_ij1 = ((ub[k] - lb[k]) * c/2) * s_func(xj_xi) * r_ij_vec[0]
                        s_ij2 = ((ub[k+1] - lb[k+1]) * c/2) * s_func(xj_xi) * r_ij_vec[1]

                        S_i[0, :] = S_i[0, :] + s_ij1
                        S_i[1, :] = S_i[1, :] + s_ij2
                S_i_total[k:k+2, :] = S_i
            Xnew = c*S_i_total.T + GbestPosition # 更新位置
            GrassHopperPositions_temp[i, :] = copy.copy(Xnew)

        X = BorderCheck(GrassHopperPositions_temp, ub, lb, pop, dim)
        fitness = Fitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)

        if(fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPosition = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPosition, Curve

