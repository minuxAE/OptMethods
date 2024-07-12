import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

"""
阻尼牛顿法, 基于Armijo非精确搜索实现
"""
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


def dampnm(x0):
    maxk=100 # 最大迭代次数
    rho = 0.55 # armijo参数
    sigma = 0.4 # armijo参数
    k = 0
    Err = []; tk = []
    eps = 1e-5

    while k < maxk:
        gk = gfun(x0)
        Gk = Hess(x0)
        dk = solve(-Gk, gk) # 求解线性方程 Gkdk = -gk
        if norm(gk) < eps:
            break
        
        # armijo搜索
        m = 0
        mk = 0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0) + sigma * rho**m * gk.T@dk:
                mk = m
                break
            m += 1
        x0 = x0 + rho**mk*dk
        err = fun(x0)
        Err.append(err)
        tk.append(k)
        k+=1
    x=x0
    val = fun(x)
    return x, val, k, Err, tk

def main():
    x0 = np.array([20, 20]).T
    x0.shape = (2, 1)
    
    x, val, k, Err, tk = dampnm(x0)
    print(x, val, k)

    plt.yscale('log')
    plt.xlabel('Iteration: k')
    plt.ylabel('Objective function: f(x_k)')
    plt.plot(tk, Err, 'b-.', lw=1)
    plt.show()

if __name__ == '__main__':
    main()
        