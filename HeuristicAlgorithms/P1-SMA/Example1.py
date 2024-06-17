"""
压力容器设计

需要求解带约束的问题
"""
import numpy as np
import matplotlib.pyplot as plt
# 定义适应度函数
def fun(X):
    x1 = X[0] # Ts
    x2 = X[1] # Th
    x3 = X[2] # R
    x4 = X[3] # L

    # 约束条件判断
    g1 = -x1+0.0193*x3
    g2 = -x2+0.00954*x3
    g3 = -np.math.pi*x3**2-4*np.math.pi*x3**3/3+1296000
    g4 = x4-240
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        fitness = 0.6224*x1*x3*x4 + 1.7781 * x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
    else:
        fitness = 1e32 # 不满足条件的情况则设置为一个超大值
    
    return fitness

# 主函数
import LibSMA

def main():
    pop = 50
    maxIter = 500
    dim = 4
    lb = np.array([0, 0, 10, 10])
    ub = np.array([100, 100, 100, 100])
    fobj = fun
    GbestScore, GbestPosition, Curve = LibSMA.SMA(pop, dim, lb, ub, maxIter, fobj)
    print('The best Fitness: ', GbestScore)
    print('Optimal Solution [Ts, Th, R, L]: ', GbestPosition)

    # 绘制适应度函数曲线
    plt.plot(Curve, 'r-', lw=1)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel('Fitness', fontsize='medium')
    plt.grid()
    plt.title('SMA', fontsize='large')
    plt.show()


if __name__ == '__main__':
    main()




