"""
Newton-Lagrangian method

求解以下最优化问题

min f(x)=exp(x1x2x3x4x5)-1/2(x1^3+x_2^3+1)^2
s.t. x1^2+x2^2+x3^2+x4^2+x5^2-10=0
     x2x3-5x4x5 = 0
     x1^3+x2^3+1=0
"""

import numpy as np
from numpy.linalg import norm, pinv, solve

###### 拉格朗日函数 ######  
def la(x,mu): 
    f = f1(x)
    h = h1(x)
    return f - mu.T * h

###### 拉格朗日函数的梯度 ######
def dla(x,mu): 
    df = df1(x)
    h = h1(x)
    dh = dh1(x)
    dlt = df - dh.T * mu
    return np.r_[dlt, -h]

###### 拉格朗日函数的Hesse阵 ######
def d2la(x,mu): 
    d2f = d2f1(x)
    d2h1,d2h2,d2h3 = d2h(x)
    return d2f - mu[0,0] * d2h1 - mu[1,0] * d2h2 - mu[2,0] * d2h3

###### 系数矩阵N(x,mu) ######
def N1(x,mu): 
    l = len(mu)
    d2l = d2la(x,mu)
    dh = dh1(x)
    zs = np.mat(np.zeros((l,l)))
    mat1 = np.c_[d2l, -dh.T]
    mat2 = np.c_[-dh, zs]
    return np.r_[mat1, mat2]

###### 目标函数f(x) ######
def f1(x): 
    s = x[0,0] * x[1,0] * x[2,0] * x[3,0] * x[4,0]
    return np.exp(s) - 0.5 * (x[0,0] ** 3 + x[1,0] ** 3 + 1) ** 2
   
###### 约束函数h(x) ######
def h1(x): 
    return np.mat([[x[0,0] ** 2 + x[1,0] ** 2 + x[2,0] ** 2 + x[3,0] ** 2 + x[4,0] ** 2 - 10],[x[1,0] * x[2,0] - 5 * x[3,0] * x[4,0]],[x[0,0] ** 3 + x[1,0] ** 3 + 1]])

###### 目标函数f(x) 的梯度 ######
def df1(x): 
    s = x[0,0] * x[1,0] * x[2,0] * x[3,0] * x[4,0]
    df0 = s / x[0,0] * np.exp(s) - 3 * (x[0,0] ** 3 + x[1,0] ** 3 + 1) * x[0,0] ** 2
    df1 = s / x[1,0] * np.exp(s) - 3 * (x[0,0] ** 3 + x[1,0] ** 3 + 1) * x[1,0] ** 2
    df2 = s / x[2,0] * np.exp(s)
    df3 = s / x[3,0] * np.exp(s)
    df4 = s / x[4,0] * np.exp(s)
    return np.mat([df0,df1,df2,df3,df4]).T

###### 约束函数h(x) 的Jacobi矩阵A(x) ######
def dh1(x): 
    return np.mat([[2 * x[0,0],2 * x[1,0],2 * x[2,0],2 * x[3,0],2 * x[4,0]],[0,x[2,0],x[1,0],- 5 * x[4,0],- 5 * x[3,0]],[3 * x[0,0] ** 2,3 * x[1,0] ** 2,0,0,0]])

###### 目标函数f(x) 的Hesse阵 ######
def d2f1(x): 
    s = x[0,0] * x[1,0] * x[2,0] * x[3,0] * x[4,0]
    return np.mat([[(s / (x[0,0])) ** 2 * np.exp(s) - 6 * x[0,0] * (x[0,0] ** 3 + x[1,0] ** 3 + 1) - 9 * x[0,0] ** 4,(1 + s) * x[2,0] * x[3,0] * x[4,0] * np.exp(s) - 9 * x[0,0] ** 2 * x[1,0] ** 2,(1 + s) * x[1,0] * x[3,0] * x[4,0] * np.exp(s),(1 + s) * x[1,0] * x[2,0] * x[4,0] * np.exp(s),(1 + s) * x[1,0] * x[2,0] * x[3,0] * np.exp(s)],[(1 + s) * x[2,0] * x[3,0] * x[4,0] * np.exp(s) - 9 * x[0,0] ** 2 * x[1,0] ** 2,(s / (x[1,0])) ** 2 * np.exp(s) - 6 * x[1,0] * (x[0,0] ** 3 + x[1,0] ** 3 + 1) - 9 * x[1,0] ** 4,(1 + s) * x[0,0] * x[3,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[2,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[2,0] * x[3,0] * np.exp(s)],[(1 + s) * x[1,0] * x[3,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[3,0] * x[4,0] * np.exp(s),s ** 2 / (x[2,0]) * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[3,0] * np.exp(s)],[(1 + s) * x[1,0] * x[2,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[2,0] * x[4,0] * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[4,0] * np.exp(s),s ** 2 / (x[3,0]) * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[2,0] * np.exp(s)],[(1 + s) * x[1,0] * x[2,0] * x[3,0] * np.exp(s),(1 + s) * x[0,0] * x[2,0] * x[3,0] * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[3,0] * np.exp(s),(1 + s) * x[0,0] * x[1,0] * x[2,0] * np.exp(s),s ** 2 / (x[4,0]) * np.exp(s)]]).T
 
###### 约束函数h(x) 的Hesse阵 ######
def d2h(x): 
    d2h1 = np.mat([[2,0,0,0,0],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]).T
    d2h2 = np.mat([[0,0,0,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,0,- 5],[0,0,0,- 5,0]]).T
    d2h3 = np.mat([[6 * x[0,0],0,0,0,0],[0,6 * x[1,0],0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]).T
    return d2h1,d2h2,d2h3

def newlagr(x0=None, mu0=None):
    # x0为初始点, mu0是乘子向量的初始值
    # x, mu分别是近似最优点以及相应的乘子
    # val是最优值, mh是约束函数的模, k是迭代次数
    maxk = 500
    n = len(x0)
    l = len(mu0)
    rho = 0.5
    gamma = 0.4
    x = x0
    mu = mu0
    k = 0
    eps = 1e-8
    while k < maxk:
        dl = dla(x, mu) # 计算Lagrangian函数梯度
        if norm(dl) < eps:
            break
        N = N1(x, mu) # 计算系数矩阵
        dz = solve(-N, dl)
        dx = dz[0: n]
        du = dz[n: n+1]
        m = 0
        mk = 0
        while m < 20:
            if norm(dla(x + rho ** m * dx, mu + rho ** m * du)) ** 2 <= (1 - gamma * rho ** m) * norm(dl) ** 2:
                mk = m
                break
            m = m+1

        x = x + rho ** mk * dx
        mu = mu + rho ** mk * du
        k = k + 1

    val = f1(x)
    mh = norm(h1(x))

    return x, mu, val, mh, k


if __name__ == '__main__':
    x0 = np.mat([-1.7,1.5,1.8, -0.6, -0.6]).T
    mu0 = np.mat([0.1,0.1,0.1]).T
    print(newlagr(x0,mu0))
