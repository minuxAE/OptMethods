"""
黏菌算法(SMA)模拟了自然界中黏菌觅食过程中的行为和形态变化
特点: 原理简单, 调节参数少, 寻优能力强, 便于实现

基本算法流程：
S1: 设定参数, 初始化种群, 计算适应度值
S2: 计算重量W和参数a
S3: 生成随机数r, 判断随机数r与参数z的大小, 如果r<z则更新个体位置, 否则更新参数p, vb, vc
S4: 计算适应度值, 更新全局最优解
"""
import numpy as np
import copy
# RandValue = np.random.random()

def initialization(pop, ub, lb, dim):
    # 初始化函数
    # pop: 种群数量
    # dim: 每个个体的维度
    # ub: 每个维度的变量上界
    # lb: 每个维度的变量下界
    # X: 输出的种群, 维度为[pop, dim]s
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]

    return X

def BorderCheck(X, ub, lb, pop, dim):
    # 边界检查函数
    # dim 个体维度大小
    # ub 个体上边界
    # lb 个体下边界
    # pop 种群数量
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
    
    return X

def Fitness(X, fun):
    # 计算种群所有个体的适应度值
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

def SortFitness(Fit):
    # 对适应度进行排序
    # input: 适应度值
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

def SortPosition(X, index):
    # 根据适应度值得大小对个体位置进行排序
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def SMA(pop, dim, lb, ub, maxIter, fun):
    # 黏菌算法
    # 输出
    # GbestScore 最优解对应得适应度值
    # GbestPosition 最优解
    # Curve 迭代曲线

    EPS = 1e-8
    z = 0.03 # 位置更新参数
    X = initialization(pop, ub, lb, dim) # 初始化种群
   
    fitness = Fitness(X, fun) # 计算适应度值
    fitness, sortIndex = SortFitness(fitness) # 对适应度进行排序
    X = SortPosition(X, sortIndex) # 种群排序
    
    GbestScore = copy.copy(fitness[0])
    GbestPosition = copy.copy(X[0, :])
    Curve = np.zeros([maxIter, 1])
    W = np.zeros([pop, dim]) # 权重矩阵
    for t in range(maxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S = bestFitness - worstFitness + EPS

        for i in range(pop):
            if i < pop/2: # 适应度排名在前一半
                W[i, :] = 1+np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / S +1)
            else:
                W[i, :] = 1-np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / S +1)

        # 惯性因子 a b
        tt = -(t / maxIter) + 1
        if tt != -1 and tt != 1:
            a = np.math.atanh(tt)
        else:
            a = 1
        
        b = 1 - t/maxIter

        # 更新位置
        for i in range(pop):
            if np.random.random() < z:
                X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
            else:
                p = np.tanh(abs(fitness[i] - GbestScore))
                vb = 2*a*np.random.random([1, dim]) - a
                vc = 2*b*np.random.random([1, dim]) - b
                
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r < p:
                        X[i, j] = GbestPosition[j] + vb[0, j]*(W[i,j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[0, j] * X[i, j]

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = Fitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        # print(X)
        if(fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPosition = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPosition, Curve

def foo1():
    def fun(x):
    # 适应度函数: 优化问题的目标函数
        fitness = np.sum(x**2)
        return fitness
    
    pop = 10
    dim = 5
    ub = np.array([5, 5, 5, 5, 5])
    lb = np.array([-5, -5, -5, -5, -5])
    X = initialization(pop, ub, lb, dim)
    # print('X:', X)

    x = np.array([1, 2])
    fitness = fun(x)
    print('fitness:', fitness)



if __name__ == '__main__':
    foo1()
