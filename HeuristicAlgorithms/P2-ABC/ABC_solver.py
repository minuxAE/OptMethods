"""
使用ABC算法求解极值
f(x, y) = x^2 + y^2
"""
import matplotlib.pyplot as plt
import numpy as np
import LibABC

# 适应度函数
def fun(X):
    return X[0]**2 + X[1]**2

def vis():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    x1 = np.arange(-10, 10, 0.2)
    x2 = np.arange(-10, 10, 0.2)
    X1, X2 = np.meshgrid(x1, x2)
    F = X1**2 + X2**2
    # 或者使用cmap = plt.cm.Blues绘制蓝色系三维图
    ax.plot_surface(X1, X2, F, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()

def main():
    pop = 50
    maxIter = 100
    dim = 2
    lb = -10 * np.ones(dim) # 下边界
    ub = 10 * np.ones(dim) # 上边界

    fobj = fun
    GbestScore, GbestPosition, Curve = LibABC.ABC(pop, dim, lb, ub, maxIter, fobj)
    print('The best fitness is: ', GbestScore)
    print('The best solution is: ', GbestPosition)

    # 绘制最优Fitness曲线
    plt.plot(Curve, 'r-', lw=1)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel('Fitness', fontsize='medium')
    plt.grid()
    plt.title('ABC', fontsize='large')
    plt.show()

if __name__ == '__main__':
    main()

