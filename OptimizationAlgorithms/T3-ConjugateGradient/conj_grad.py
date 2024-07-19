"""
共轭梯度法实现程序
"""
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

# 目标函数
def fun(x): 
    f = 100 * (x[0][0] ** 2 - x[1][0]) ** 2 + (x[0][0] - 1) ** 2
    return f

# 梯度函数
def gfun(x): 
    arr = np.array([400 * x[0][0] * (x[0][0] ** 2 - x[1][0]) + 2 * (x[0][0] - 1), -200 * (x[0][0] ** 2 - x[1][0])])
    g = np.transpose(arr)
    g.shape = (2, 1)
    return g

def frcg(x0):
    """
    使用共轭梯度法求解无约束优化问题: min f(x)
    x0 是初始点, fun, gfun分别为目标函数和梯度
    x, val分别是近似最优点和最优值, k是迭代次数
    """
    maxk = 5000
    rho = 0.6 # armijo参数
    sigma = 0.4 # armijo参数
    k = 0
    Err = []
    tk = []
    eps = 1e-4
    n = len(x0)
    while k < maxk:
        g = gfun(x0) # 计算在x0的梯度
        itern = k - (n+1)*int(np.floor(k / (n+1)))
        itern = itern + 1

        if itern == 1:
            d = -g # 选择使用负梯度方向
        else:
            beta = (g.T@g) / (g0.T@g0)
            d = -g + beta * d0
            gd = g.T@d
            if gd >= 0.0:
                d = -g
            
        if norm(g) < eps:
            break

        m = 0
        mk = 0
        while m < 20:
            if fun(x0 + rho**m*d) < fun(x0)+sigma*rho**m*g.T@d:
                mk = m
                break
            m = m+1
        
        x0 = x0 + rho**mk*d
        val = fun(x0)
        Err.append(val)
        tk.append(k)
        g0 = g
        d0 = d
        k = k+1

    x = x0
    val = fun(x)
    return x, val, k, Err, tk


def main():
    ## 测试不同初始点对算法的影响
    x0 = np.array([0, 0]).T
    x0.shape = (2, 1)
    # print(frcg(x0))

    x1 = np.array([0.5, 0.5]).T
    x1.shape = (2, 1)

    x2 = np.array([1.2, 1]).T
    x2.shape = (2, 1)

    x3 = np.array([-1.2, 1]).T
    x3.shape = (2, 1)

    x4 = np.array([-1.2, -1]).T
    x4.shape = (2, 1)
    print(frcg(x4))

    x, val, k, Err, tk = frcg(x0)
    print(x, val, k)

    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel('Obj function')
    plt.plot(tk, Err, 'b-.')
    plt.show()


if __name__ == '__main__':
    main()
