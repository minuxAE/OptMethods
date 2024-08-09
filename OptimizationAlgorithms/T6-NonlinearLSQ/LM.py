"""
使用LM算法求解F(x)=0
"""
import numpy as np
from numpy.linalg import norm, inv, solve

"""
求解非线性方程组
x1 - 0.7sin(x1) - 0.2cos(x2) = 0
x2 - 0.7cos(x1) + 0.2sin(x2) = 0
"""
def Fk(x=None):
    y1 = x[0, 0] - 0.7 * np.sin(x[0, 0]) - 0.2 * np.cos(x[1, 0])
    y2 = x[1, 0] - 0.7 * np.cos(x[0, 0]) + 0.2 * np.sin(x[1, 0])
    return np.mat([[y1], [y2]])

def JFk(x=None):
    JF = np.mat([
        [1-0.7*np.cos(x[0, 0]), 0.2*np.sin(x[1, 0])],
        [0.7*np.sin(x[0, 0]), 1+0.2*np.cos(x[1, 0])]
    ])
    return JF


def lmm(x0):
    maxk = 100
    rho = 0.55
    sigma = 0.4
    muk = norm(Fk(x0))
    k = 0
    eps = 1e-6
    n = len(x0)

    while k < maxk:
        fk = Fk(x0)
        jfk = JFk(x0)
        gk = jfk.T * fk
        dk = solve(-(jfk.T * jfk + muk * np.eye(n)), gk)

        if norm(gk) < eps:
            break

        m = 0
        mk = 0

        while m < 20:
            newf = 0.5 * norm(Fk(x0 + rho**m*dk))**2
            oldf = 0.5 * norm(Fk(x0)) ** 2
            if newf < oldf + sigma * rho ** m * gk.T * dk:
                mk = m
                break

            m = m + 1

        x0 = x0 + rho**mk*dk
        muk = norm(Fk(x0))
        k = k + 1

    x = x0
    val = 0.5 * muk ** 2
    return x, val, k
    
if __name__ == '__main__':
    x0 = np.array([2, 1]).T
    x0.shape = (2, 1)
    xk, val, k = lmm(x0)

    print(xk, val, k)
