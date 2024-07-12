"""
修正牛顿法
"""
import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

def fun(x):
    # 目标函数
    f = 100 * (x[0][0]**2-x[1][0])**2 + (x[0][0]-1)**2
    return f

def gfun(x):
    # 目标函数的梯度函数
    arr = np.array([400*x[0][0]*(x[0][0]**2-x[1][0])+2*(x[0][0]-1), -200*(x[0][0]**2-x[1][0])])
    g = arr.T
    g.shape=(2, 1)
    return g

def Hess(x):
    # Hess矩阵
    n = len(x)
    He = np.zeros((n, n))
    He = np.array([
        [1200*x[0][0]**2-400*x[1][0]+2, -400*x[0][0]],
        [-400*x[0][0], 200]
    ])

    return He

def revisenm(x0):
    # 求解问题 min f(x)
    n = len(x0)
    maxk = 150
    rho = 0.55
    sigma = 0.4
    tau = 0.0
    k = 0
    Err = []
    tk = []
    eps = 1e-5
    while k < maxk:
        gk = gfun(x0)
        muk = norm(gk)**(1+tau)
        Gk = Hess(x0)
        Ak = Gk + muk * np.eye(n)
        dk = solve(-Ak, gk)
        if norm(gk) < eps:
            break

        m = 0
        mk = 0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0) + sigma*rho**m*gk.T@dk:
                mk = m
                break
            m = m+1

        x0 = x0 + rho ** mk * dk
        err = fun(x0)
        Err.append(err)
        tk.append(k)
        k = k+1

    x = x0
    val = fun(x)

    return x, val, k, Err, tk

from dampedNewton import dampnm

def main():
    x0 = np.array([0, 0]).T
    x0.shape = (2, 1)
    
    x1, val1, k1, Err1, tk1 = dampnm(x0)
    x2, val2, k2, Err2, tk2 = revisenm(x0)

    plt.yscale('log')
    plt.xlabel('Iteration: k')
    plt.ylabel('Objective function: f(x_k)')
    plt.plot(tk1, Err1, 'r-o', lw=1, label='Damped Newton')
    plt.plot(tk2, Err2, 'b-.', lw=1, label='Revised Newton')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()