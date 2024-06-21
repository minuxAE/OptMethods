import LibSMA
import matplotlib.pyplot as plt
import numpy as np

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
   vis()
   # main()