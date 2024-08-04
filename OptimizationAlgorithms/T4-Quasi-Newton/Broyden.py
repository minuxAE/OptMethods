"""
Broyden族算法程序
"""
import numpy as np
from numpy.linalg import norm, solve, inv
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

def Hess(x):
    # Hess矩阵
    n = len(x)
    He = np.zeros((n, n))
    He = np.array([
        [1200*x[0][0]**2-400*x[1][0]+2, -400*x[0][0]],
        [-400*x[0][0], 200]
    ])

    return He

def broyden(x0):
    """
    用Broyden族算法求解无约束优化问题
    """
    maxk = 1e4
    rho = 0.55
    sigma = 0.4
    eps = 1e-5
    phi = 0.5
    k = 0
    Err = []
    tk = []
    n = len(x0)
    Hk = inv(Hess(x0))

    while k < maxk:
        gk = gfun(x0)
        if norm(gk) < eps:
            break
        dk = -Hk@gk
        
        m=0
        mk=0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*(gk.T)@dk:
                mk = m
                break
            m = m+1

        x = x0 + rho ** mk * dk
        sk = x - x0
        yk = gfun(x) - gk
        Hy = Hk@yk
        sy = (sk.T)@yk
        yHy = (yk.T)@Hk@yk

        if sy < 0.2 * yHy:
            theta = 0.8 * yHy/(yHy - sy)
            sk = theta * sk + (1-theta)*Hy
            sy = 0.2 * yHy
            
        vk = np.sqrt(np.abs(yHy))*(sk/sy - Hy/yHy)
        Hk = Hk-(Hy@Hy.T) / yHy + (sk@sk.T) / sy + phi*(vk@vk.T)

        val = fun(x0)
        Err.append(val)
        tk.append(k)
        k = k+1
        x0 = x
    
    val = fun(x0)
    return x, val, k, Err, tk


from SR1 import sr1
from BFGS import bfgs
from DFP import dfp

def main():
    x0 = np.array([10, 10]).T
    x0.shape = (2, 1)

    x1, val1, k1, Err1, tk1 = sr1(x0)
    x2, val2, k2, Err2, tk2 = bfgs(x0)
    x3, val3, k3, Err3, tk3 = dfp(x0)
    x4, val4, k4, Err4, tk4 = broyden(x0)

    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel('Objective function f(xk)')

    plt.plot(tk1, Err1, 'b-.', label='SR1')
    plt.plot(tk2, Err2, 'r-.', label='BFGS')
    plt.plot(tk3, Err3, 'y-.', label='DFP')
    plt.plot(tk4, Err4, 'g-.', label='Broyden')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



