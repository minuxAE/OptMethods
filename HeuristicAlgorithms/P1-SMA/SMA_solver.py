import LibSMA
import matplotlib.pyplot as plt
import numpy as np

# 适应度函数
def fun(X):
    return X[0]**2 + X[1]**2

# 主函数
def main():
    pop = 50
    maxIter = 100
    dim = 2
    lb = -10 * np.ones(dim)
    ub = 10 * np.ones(dim)
    fobj = fun
    GbestScore, GbestPosition, Curve = LibSMA.SMA(pop, dim, lb, ub, maxIter, fobj)
    print('The best Fitness is: ', GbestScore)
    print('[x1, x2] is: ', GbestPosition)

    # 绘制适应度曲线
    plt.plot(Curve, 'r-', lw=1)
    plt.show()


if __name__ == '__main__':
    main()