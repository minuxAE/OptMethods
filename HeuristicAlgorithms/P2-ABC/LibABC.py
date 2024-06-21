import numpy as np
import copy as copy
"""
种群初始化函数: initialization
"""
def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j]=(ub[j] - lb[j])*np.random.random()+lb[j] # [lb, ub]之间的随机数生成

    return X

"""
适应度函数编写
"""
def fun(x):
    fitness = np.sum(x**2)
    return fitness

"""
边界检查和约束函数
"""
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

"""
轮盘赌策略
"""
def RouletteWheelSelection(P):
    C = np.cumsum(P)
    r = np.random.random() * C[-1]
    # 定义选择阈值，将随机概率与输入向量P的总和的乘积作为阈值
    out = 0
    for i in range(P.shape[0]):
        if r < C[i]:
            out = i
            break
    return out

"""
适应度函数
"""
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

def ABC(pop, dim, lb, ub, maxIter, fun):
    L = round(0.6*dim*pop) # limit参数
    C = np.zeros([pop, 1]) # 计数器
    nOnlooker = pop # leader的数量

    X = initialization(pop, ub, lb, dim) # 初始化种群
    fitness = Fitness(X, fun) # 计算适应度值
    fitness, sortIndex = SortFitness(fitness) # 对适应度值进行排序
    X = SortPosition(X, sortIndex)

    GbestScore = copy.copy(fitness[0])
    GbestPosition = np.zeros([1, dim])
    GbestPosition[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])
    Xnew = np.zeros([pop, dim])
    fitnessNew = copy.copy(fitness)

    for t in range(maxIter):
        # leader搜索
        for i in range(pop):
            k = np.random.randint(pop) # 随机选择一个个体
            while(k==i):
                k = np.random.randint(pop) # 要求不能是相同的个体
            phi = (2*np.random.random([1, dim])-1)
            Xnew[i, :] = X[i, :] + phi * (X[i, :] - X[k, :]) # 根据ABC算法更新位置
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim) # 进行边界检查
        fitnessNew = Fitness(Xnew, fun) # 计算适应度函数

        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i]+1
        
        # 计算适应度值
        F = np.zeros([pop, 1])
        MeanCost = np.mean(fitness)
        for i in range(pop):
            F[i] = np.exp(-fitness[i]/MeanCost)
        P = F / sum(F)

        # 侦察bee搜索
        for m in range(nOnlooker):
            i = RouletteWheelSelection(P)
            k = np.random.randint(pop)
            while(k==i):
                k = np.random.randint(pop)
            phi = (2 * np.random.random([1, dim]) - 1)
            Xnew[i, :] = X[i, :] + phi * (X[i, :] - X[k, :]) # update location
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = Fitness(Xnew, fun) # 计算适应度值
        
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                X[i, :] = copy.copy(Xnew[i, :]) # 当前位置具有更好的适应度值，替换原始位置
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i] + 1 # 如果位置没有更新，累加器+1
        
        # 判断limit条件
        for i in range(pop):
            if C[i] >= L:
                for j in range(dim):
                    X[i, j] = np.random.random() * (ub[j] - lb[j]) + lb[j]
                    C[i] = 0
        
        fitness = Fitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0, :] = copy.copy(X[0, :])
        
        Curve[t] = GbestScore

    return GbestScore, GbestPosition, Curve