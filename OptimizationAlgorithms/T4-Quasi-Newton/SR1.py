import numpy as np
from numpy.linalg import norm
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


"""
用SR1算法求解无约束优化问题: min f(x)
"""
def sr1(x0):
    maxk = 500
    rho = 0.55
    sigma = 0.4
    eps = 1e-5
    k = 0
    Err = []
    tk = []
    n = len(x0)
    Hk = np.eye(n)
    while k < maxk:
        gk = gfun(x0)
        dk = -Hk@gk
        if norm(gk) < eps:
            break

        m = 0
        mk = 0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*gk.T@dk:
                mk = m
                break

            m = m+1

        x = x0 + rho**mk*dk
        val = fun(x0)
        Err.append(val)
        tk.append(k)
        sk = x-x0
        yk = gfun(x) - gk
        Hk = Hk+(sk-Hk@yk)@((sk-Hk@yk).T) / ((sk-Hk@yk).T@yk)
        k = k+1
        x0 = x

    x = x0
    val = fun(x0)
    return x, val, k, Err, tk


def main():
    x0 = np.array([10, 10]).T
    x0.shape = (2, 1)
    x, val, k, Err, tk = sr1(x0)

    plt.xticks(range(0, k, 10))
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Obj: f(x)')
    plt.plot(tk, Err, 'b-.')
    plt.show()

if __name__ == '__main__':
    main()



