import numpy as np
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from SR1 import sr1

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
BFGS算法求解无约束优化问题
"""
def bfgs(x0):
    maxk = 500
    rho=0.55
    sigma=0.4
    eps = 1e-5
    k=0
    Err = []
    tk = []
    n = len(x0)
    Bk = np.eye(n)

    while k < maxk:
        gk = gfun(x0)
        if norm(gk) < eps:
            break
        dk = solve(-Bk, gk)
        # 使用armijo搜索步长
        m = 0
        mk = 0
        while m<20:
            newf = fun(x0+rho**m*dk)
            oldf = fun(x0)
            if newf < oldf + sigma * rho ** m * gk.T@dk:
                mk = m
                break
            m = m+1
        
        x = x0 + rho ** mk * dk
        val = fun(x0)
        Err.append(val)
        tk.append(k)
        sk = x - x0
        yk = gfun(x) - gk
        if yk.T@sk > 0:
            Bk = Bk - (Bk@sk@sk.T@Bk) / (sk.T@Bk@sk) + (yk@yk.T) / (yk.T@sk)

        k = k+1
        x0 = x

    val = fun(x0)
    return x, val, k, Err, tk 

def main():
    x0 = np.array([10, 10]).T # 设定初始点
    x0.shape = (2, 1)
    x1, val1, k1, Err1, tk1 = bfgs(x0)
    x2, val2, k2, Err2, tk2 = sr1(x0)
    plt.yscale('log')
    plt.xlabel('Iteration K')
    plt.ylabel('Obj f(x_k)')
    plt.plot(tk1, Err1, 'r-.', label='BFGS')
    plt.plot(tk2, Err2, 'b-.', label='SR1')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()



