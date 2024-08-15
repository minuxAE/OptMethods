import numpy as np
from numpy.linalg import norm, solve

"""
min f(x) = (x1-2)^2 + (x2-1)^2
s.t. x1 - 2x2 + 1=0
     0.25x1^2 + x2^2 <= 1
"""

# 目标函数
def f1(x):
    f = (x[0, 0] - 2.0)**2 + (x[1, 0]-1.0)**2
    return f

# 等式约束
def h1(x):
    he = x[0, 0] - 2.0 * x[1, 0] + 1.0
    return np.mat([he])

# 不等式约束
def g1(x):
    g1 = -0.25 * x[0, 0]**2 - x[1, 0]**2 + 1
    return np.mat([g1])

# 目标函数的梯度
def df1(x):
    return np.mat([2.0 * (x[0, 0] - 2.0), 2.0 * (x[1, 0]-1.0)]).T

# 等式约束函数的Jacobi矩阵
def dh1(x):
    return np.mat([1.0, -2.0]).T

# 不等式约束向量函数的Jacobi矩阵
def dg1(x):
    return np.mat([-0.5 * x[0, 0], -2.0 * x[1, 0]]).T

# 增广拉格朗日函数
def mpsi(x,mu,lambda_,sigma): 
    f = f1(x)
    he = h1(x)
    gi = g1(x)
    l = len(he)
    m = len(gi)
    psi = f
    s1 = 0.0
    for i in range(l):
        psi = psi- he[i] * mu[i]
        s1 = s1 + he[i] ** 2

    psi = psi + 0.5 * sigma * s1
    s2 = 0.0
    for i in range(m):
        s3 = max(0.0, lambda_[i] - sigma * gi[i]);
        s2 = s2 + s3 ** 2 - lambda_[i] ** 2;
        
    psi = psi + s2 / (2.0 * sigma)
    return psi[0,0]

# 增广拉格朗日函数的梯度
def dmpsi(x,mu,lambda_,sigma): 
    dpsi = df1(x)
    he = h1(x)
    gi = g1(x)
    dhe = dh1(x)
    dgi = dg1(x)
    l = len(he)
    m = len(gi)
    for i in range(l):
        dpsi = dpsi + (sigma * he[i,0] - mu[i,0]) * dhe[:,i]
    for i in range(m):
        dpsi = dpsi + (sigma * gi[i,0] - lambda_[i,0]) * dgi[:,i]

    return dpsi

def bfgs(x0,mu,lambda_,sigma2): 
    '''
    功能: 用BFGS算法求解无约束问题: min f(x)
    输入: x0是初始点, fun, gfun分别是目标函数及其梯度;
    varargin是输入的可变参数变量, 简单调用bfgs时可以忽略它,
    但若其它程序循环调用该程序时将发挥重要的作用
    输出: x, val分别是近似最优点和最优值, k是迭代次数.
    '''
    maxk = 500
    
    rho = 0.55
    sigma = 0.4
    epsilon = 1e-05
    k = 0
    n = len(x0)
    Bk = np.eye(n)
    
    while (k < maxk):

        gk = dmpsi(x0,mu,lambda_,sigma2)
        if (norm(gk) < epsilon):
            break
        dk = solve(-Bk,gk)
        m = 0
        mk = 0
        while (m < 20):
            newf = mpsi(x0 + rho ** m * dk,mu,lambda_,sigma2)
            oldf = mpsi(x0,mu,lambda_,sigma2)
            if (newf < oldf + sigma * rho ** m * gk.T * dk):
                mk = m
                break
            m = m + 1

        x = x0 + rho ** mk * dk
        sk = x - x0
        yk = dmpsi(x,mu,lambda_,sigma2) - gk
        if (yk.T * sk > 0):
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)
        k = k + 1
        x0 = x
    
    val = mpsi(x0,mu,lambda_,sigma2)
    return x,val,k

# PHR算法
def multphr(x0=None):
    """
    使用乘子法求解一般约束优化问题 min f(x) s.t. h(x)=0, g(x)>=0
    """
    maxk = 500
    sigma = 2.0 # 罚因子
    eta = 2.0
    theta = 0.8
    k = 0 # 外部迭代
    ink = 0 # 内部迭代
    eps = 1e-5
    x = x0
    he = h1(x)
    gi = g1(x)
    n = len(x)
    l = len(he)
    m = len(gi)

    # 选取乘子向量的初始值
    mu = 0.1 * np.mat(np.ones((1, 1)))
    lam = 0.1 * np.mat(np.ones((m, 1)))

    # 终止准测参数
    betak = 10
    betaold = 10

    while betak > eps and k < maxk:
        # 使用BFGS算法求解无约束子问题
        x, v, ik = bfgs(x0, mu, lam, sigma)
        ink = ink + ik
        he = h1(x)
        gi = g1(x)

        betak = 0.0
        for i in range(l):
            betak = betak + he[i] ** 2

        for i in range(m):
            temp = min(gi[i], lam[i] / sigma)
            betak = betak + temp ** 2

        betak = np.sqrt(betak)

        if betak > eps:
            if k>=2 and betak > theta * betaold:
                sigma = eta * sigma
            
            # 更新乘子向量
            for i in range(l):
                mu[i] = mu[i] - sigma * he[i]

            for i in range(m):
                lam[i] = max(0, lam[i] - sigma * gi[i])

        k = k+1
        betaold = betak
        x0 = x

    f = f1(x)
    return x, mu, lam, f, k, ink, betak


if __name__ == '__main__':
    x0 = np.mat([3, 3]).T
    x, mu, lam, f, k, ink, betak = multphr(x0)
    print(x, mu, lam, f, k, ink, betak) # (0.8229, 0.9114) f*=1.3934




