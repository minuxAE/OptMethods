import numpy as np
from numpy.linalg import norm
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

def grad(x0):
    # 使用梯度下降法求解无约束优化问题
    maxk = 5000 # 最大迭代次数
    rho = 0.5 # Armijo搜索参数 (0, 1)
    sigma = 0.4 # Armijo搜索参数 (0, 0.5)
    k = 0
    Err = []; tk = []
    eps = 1e-5
    while k<maxk:
        g = gfun(x0)
        d = -g # 选择负梯度方向
        if norm(d) < eps: # 如果当前梯度几乎为0, 则跳出循环
            break
        
        # 使用Armijo非精确条件设定步长
        m = 0
        mk = 0 # mk是满足下列不等式的最小整数
        while m<20:
            if fun(x0+rho**m*d) < fun(x0)+sigma*rho**m*g.T@d:
                mk = m
                break
            m += 1
        x0 = x0 + rho**mk*d # 更新迭代点
        err = fun(x0)
        Err.append(err)
        tk.append(k)
        k+=1

    x=x0
    val = fun(x0)
    return x, val, k, Err, tk

def main():
    x0 = np.array([0, 0]).T
    x0.shape = (2, 1)
    # print(gfun(x0))
    
    x, val, k, Err, tk = grad(x0)
    print(x, val, k)

    plt.yscale('log')
    plt.xlabel('Iteration: k')
    plt.ylabel('Objective function: f(x_k)')
    plt.plot(tk, Err, 'b-.', lw=1)
    plt.show()

if __name__ == '__main__':
    main()


