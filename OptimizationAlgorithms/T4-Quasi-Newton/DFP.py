import numpy as np
from numpy.linalg import norm, solve, inv
import matplotlib.pyplot as plt
from BFGS import bfgs

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

def dfp(x0):
    maxk = 1e4
    rho = 0.55
    sigma = 0.4
    eps = 1e-5
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
        
        # armijo搜索步长
        m = 0
        mk = 0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*gk.T@dk:
                mk = m
                break
            m = m+1

        x = x0+rho**mk*dk
        val = fun(x0)
        Err.append(val)
        tk.append(k)
        sk = x-x0
        yk = gfun(x) - gk

        if (sk.T)@yk > 0:
            Hk = Hk - (Hk@yk@(yk.T)@Hk) / ((yk.T)@Hk@yk) + (sk@(sk.T)) / ((sk.T)@yk)
        
        k = k+1
        x0 = x
    val = fun(x0)
    return x, val, k, Err, tk

if __name__ == '__main__':
    x0 = np.array([10, 10]).T
    x0.shape = (2, 1)
    x1, val1, k1, Err1, tk1 = bfgs(x0)
    x2, val2, k2, Err2, tk2 = dfp(x0)

    plt.yscale('log')
    plt.ylabel('Obj f(xk)')
    plt.plot(tk1, Err1, 'r-.', label='BFGS')
    plt.plot(tk2, Err2, 'b-.', label='DFP')
    plt.legend()
    plt.show()


        