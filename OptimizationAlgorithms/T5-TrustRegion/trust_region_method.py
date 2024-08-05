"""
信赖域方法程序
"""
from trust_region_sub import trustsq

import numpy as np
from numpy.linalg import norm, inv

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

# 信赖域子问题目标函数文件
def qk(x, d):
    gk = gfun(x)
    Bk = Hess(x)
    qd = gk.T * d + 0.5 * d.T * Bk * d
    return qd

def trustm(x0):
    """
    使用Newton型信赖域方法求解无约束优化问题 min f(x)
    """
    n = len(x0)
    x = x0
    dta = 1
    eta1 = 0.1
    eta2 = 0.75
    dtabar = 2.0
    tau1 = 0.5
    tau2 = 2.0
    eps = 1e-6
    k = 0
    Bk = Hess(x)

    while k < 50:
        gk = gfun(x)
        if norm(gk) < eps:
            break
        # 求解子问题
        d, val, lam, ik = trustsq(gk, Bk, dta)
        print(d, val, lam, ik)
        deltaq = -qk(x, d)
        deltaf = fun(x) - fun(x+d)
        rk = deltaf / deltaq
        print('rk=', rk, 'eta1=', eta1)

        if rk <= eta1:
            dta = tau1 * dta
        else:
            if rk >= eta2 and norm(d) == dta:
                dta = min(tau2 * dta, dtabar)
            else:
                dta = dta
        if rk > eta1:
            x = x + d
            Bk = Hess(x)

        k = k+1
    xk = x
    val = fun(xk)
    return xk, val, k

if __name__ == '__main__':
    x0 = np.array([2, 1]).T
    x0.shape = (2, 1)
    xk, val, k = trustm(x0)

    print(xk, val, k)
