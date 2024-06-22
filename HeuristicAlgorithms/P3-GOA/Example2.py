"""
三杆设计
"""
import numpy as np
import matplotlib.pyplot as plt
import LibGOA

# 适应度函数
def fun(X):
    x1 = X[0]
    x2 = X[1]
    l = 100
    P = 2
    sigma = 2
    # 约束条件判断
    g1 = (np.sqrt(2)*x1+x2)*P / (np.sqrt(2)*x1**2+2*x1*x2)-sigma
    g2 = x2*P / (np.sqrt(2)*x1**2+2*x1*x2) - sigma
    g3 = P / (np.sqrt(2)*x2+x1)-sigma

    if g1<=0 and g2<=0 and g3<=0:
        # 满足约束条件，计算适应度值
        fitness = (2*np.sqrt(2)*x1+x2)*l
    else:
        fitness = 1e32
    return fitness

def main():
    pop = 30
    maxIter = 100
    dim = 2
    lb = np.array([0.001, 0.001])
    ub = np.array([1, 1])

    fobj = fun
    GbestScore, GbestPosition, Curve = LibGOA.GOA(pop, dim, lb, ub, maxIter, fobj)
    print('Fitness: ', GbestScore)
    print('Optimal Solution: ', GbestPosition)

    plt.figure()
    plt.plot(Curve, 'r-', lw=1)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel('Fitness', fontsize='medium')
    plt.grid()
    plt.title('GOA', fontsize='large')
    plt.show()

if __name__ == '__main__':
    main()
