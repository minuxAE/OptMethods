"""
混合约束优化问题的SQP方法
"""
import numpy as np
import QPsubSQP as qs
from numpy.linalg import norm, pinv, solve

"""
求解非线性规划问题

min f(x)=-pix1^2x2
s.t. pi * x1 * x2 - pi * x1^2 - 150 = 0
     x1 >= 0, x2 >= 0
"""

######### l1精确价值函数 #########
def phi1(x = None,sigma = None): 
    f = f1(x)
    h,g = cons(x)
    gn = np.maximum(-g,0)
    l0 = np.size(h)
    m0 = np.size(g)
    if (l0 == 0):
        p = f + 1.0 / sigma * norm(gn,1)
    
    if (m0 == 0):
        p = f + 1.0 / sigma * norm(h,1)
    
    if (l0 > 0 and m0 > 0):
        p = f + 1.0 / sigma * (norm(h,1) + norm(gn,1))
    
    return p

######### 价值函数的方向导数 #########
def dphi1(x = None,sigma = None,d = None): 
    df = df1(x)
    h,g = cons(x)
    gn = np.maximum(-g,0)
    l0 = np.size(h)
    m0 = np.size(g)
    if (l0 == 0):
        dp = df.T * d - 1.0 / sigma * norm(gn,1)
    
    if (m0 == 0):
        dp = df.T * d - 1.0 / sigma * norm(h,1)
    
    if (l0 > 0 and m0 > 0):
        dp = df.T * d - 1.0 / sigma * (norm(h,1) + norm(gn,1))
    
    return dp


######### 拉格朗日函数L(x,mu) #############
def la(x = None,mu = None,lam = None): 
    f = f1(x)
    h,g = cons(x)
    l0 = np.size(h)
    m0 = np.size(g)
    if (l0 == 0):
        l = f - lam * g
    
    if (m0 == 0):
        l = f - mu.T * h
    
    if (l0 > 0 and m0 > 0):
        l = f - mu.T * h - lam.T * g
    
    return l

######### 拉格朗日函数的梯度 #############
def dlax(x = None,mu = None,lam = None): 
    df = df1(x)
    Ae,Ai = dcons(x)
    m1,m2 = Ai.shape
    l1,l2 = Ae.shape
    if (l1 == 0):
        dl = df - Ai.T * lam
    
    if (m1 == 0):
        dl = df - Ae.T * mu
    
    if (l1 > 0 and m1 > 0):
        dl = df - Ae.T * mu - Ai.T * lam
    
    return dl


######## 目标函数f(x) ###########
def f1(x = None): 
    f = - np.pi * x[0,0] ** 2 * x[1,0]
    return f

####### 目标函数f(x)的梯度 ########
def df1(x = None): 
    df = np.mat([-2 * np.pi * x[0,0] * x[1,0],- np.pi * x[0,0] ** 2]).T
    return df

#########约束函数##############
def cons(x = None): 
    h = np.mat([np.pi * x[0,0] * x[1,0] + np.pi * x[0,0] ** 2 - 150])
    g = np.mat([[x[0,0]],[x[1,0]]])
    return h,g

########约束函数Jacobi矩阵######## 
def dcons(x = None): 
    dh = np.mat([np.pi * x[1,0] + 2 * np.pi * x[0,0],np.pi * x[0,0]])
    dg = np.mat([[1,0],[0,1]])
    return dh,dg

def sqpm(x0 = None,mu0 = None,lam0 = None): 
    '''
    功能: 用基于拉格朗日函数Hesse阵的SQP方法求解约束优化问题:
    min f(x) s.t. h_i(x)=0, i=1,..., l.
    输入: x0是初始点, mu0是乘子向量的初始值
    输出: x, mu分别是近似最优点及相应的乘子,
    val是最优值, mh是约束函数的模, k是迭代次数.
    '''
    maxk = 100
    
    n = np.size(x0)
    l = np.size(mu0)
    m = np.size(lam0)

    rho = 0.5
    eta = 0.1
    B0 = np.mat(np.eye(n))
    x = x0
    mu = mu0
    lam = lam0
    Bk = B0
    sigma = 0.8
    epsilon1 = 1e-06
    epsilon2 = 1e-05
    hk,gk = cons(x)
    dfk = df1(x)
    Ae,Ai = dcons(x)
    if np.size(Ae) > 0:
        Ak = np.r_[Ae, Ai]
    else:
         Ak = Ai
    k = 0
    while (k < maxk):

        dk,mu,lam,_,_ = qs.qpsubp(dfk,Bk,Ae,hk,Ai,gk)

        if np.size(hk) > 0:
            mp1 = norm(hk,1) + norm(np.maximum(-gk,0),1)
        else:
            mp1 = norm(np.maximum(-gk,0),1)
        if (norm(dk,1) < epsilon1 and mp1 < epsilon2):
            break
        deta = 0.05
        
        if np.size(mu) > 0:
            tau = max(norm(mu, np.inf), norm(lam, np.inf))
        else:
            tau = norm(lam, np.inf)
        
        if (sigma * (tau + deta) < 1):
            sigma = sigma
        else:
            sigma = 1.0 / (tau + 2 * deta)
        im = 0
        while (im <= 20):
            temp = eta * rho ** im * dphi1(x,sigma,dk)
            if (phi1(x + rho ** im * dk, sigma) - phi1(x, sigma) < temp):
                mk = im
                break
            im = im + 1
            if (im == 20):
                mk = 10

        alpha = rho ** mk
        x1 = x + alpha * dk
        hk,gk = cons(x1)
        dfk = df1(x1)
        Ae,Ai = dcons(x1)
        if np.size(Ae) > 0:
            Ak = np.r_[Ae, Ai]
        else:
             Ak = Ai
        lamu = pinv(Ak).T * dfk
        if (l > 0 and m > 0):
            mu = lamu[0:l]
            lam = lamu[l:l + m]
        if (l == 0):
            mu = []
            lam = lamu
        if (m == 0):
            mu = lamu
            lam = []
        sk = alpha * dk
        yk = dlax(x1,mu,lam) - dlax(x,mu,lam)
        if (sk.T * yk > 0.2 * sk.T * Bk * sk):
            theta = 1
        else:
            theta = (0.8 * sk.T * Bk * sk / (sk.T * Bk * sk - sk.T * yk))[0,0]
        zk = theta * yk + (1 - theta) * Bk * sk
        Bk = Bk + zk * zk.T / (sk.T * zk) - (Bk * sk) * (Bk * sk).T / (sk.T * Bk * sk)
        x = x1
        k = k + 1

    
    val = f1(x)
    return x,mu,lam,val,k


def main1():
    x0 = np.mat([3,3]).T
    mu0 = np.mat([0]).T
    lam0 = np.mat([0,0]).T
    x,mu,lam,val,k = sqpm(x0,mu0,lam0)

    print(x,mu,lam,val,k)

"""
求解非线性规划问题
min f(x)=x1^2+x2^2-16x1-10x2
s.t. -x1^2+6x1-4x2+11 >= 0
     x1x2-3x2-exp(x1-3)+1 >= 0
     x1 >= 0, x2 >= 0
"""

######## 目标函数f(x) ###########
def f1(x = None): 
    f = x[0,0] ** 2 + x[1,0] ** 2 - 16 * x[0,0] - 10 * x[1,0]
    return f

####### 目标函数f(x)的梯度 ########
def df1(x = None): 
    df = np.mat([[2 * x[0,0] - 16],[2 * x[1,0] - 10]])
    return df

#########约束函数##############
def cons(x = None): 
    h = np.mat([])
    g = np.mat([[- x[0,0] ** 2 + 6 * x[0,0] - 4 * x[1,0] + 11],[x[0,0] * x[1,0] - 3 * x[1,0] - np.exp(x[0,0] - 3) + 1],[x[0,0]],[x[1,0]]])
    return h,g

########约束函数Jacobi矩阵######## 
def dcons(x = None): 
    dh = np.mat([]).T
    dg = np.mat([[- 2 * x[0,0] + 6,- 4],[x[1,0] - np.exp(x[0,0] - 3),x[0,0] - 3],[1,0],[0,1]])
    return dh,dg

def main2():
    x0 = np.mat([4,4]).T
    mu0 = np.mat([]).T
    lam0 = np.mat([0,0,0,0]).T
    x,mu,lam,val,k = sqpm(x0,mu0,lam0)
    print(x,mu,lam,val,k)



if __name__ == '__main__':
    main2()
    # main1()