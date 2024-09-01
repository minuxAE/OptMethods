"""
使用光滑牛顿法求解二次规划子问题
"""
import numpy as np
from numpy.linalg import norm, pinv, solve

"""
求解二次规划子问题
min qk(d) = 0.5 * d*Bk * d + dfk' * d
s.t. hk + Ae * d = 0, gk + Ai * d >= 0

dfk是xk处的梯度, Bk是第k次近似的Hesse矩阵, Ae, hk线性等式约束
Ai, gk是线性不等式约束
"""

def phi(ep = None,a = None,b = None): 
    p = a + b - np.sqrt(a ** 2 + b ** 2 + 2 * ep ** 2)
    return p

def dah(ep = None,d = None,mu = None,lam = None,dfk = None,Bk = None,Ae = None,hk = None,Ai = None,gk = None): 
    n = np.size(dfk)
    l = np.size(hk)
    m = np.size(gk)

    dh = np.mat(np.zeros((n + l + m + 1,1)))
    dh[0] = ep
    if (l > 0 and m > 0):
        dh[1:n+1] = Bk * d - Ae.T * mu - Ai.T * lam + dfk
        dh[n + 1:n + l+1] = hk + Ae * d
        for i in range(m):
            dh[n+l+1+i] = phi(ep,lam[i],gk[i] + Ai[i] * d)
    
    if (l == 0):
        dh[1:n+1] = Bk * d - Ai.T * lam + dfk
        for i in range(m):
            dh[n+1+i] = phi(ep,lam[i],gk[i]+Ai[i]*d)
    
    if (m == 0):
        dh[1:n+1] = Bk * d - Ae.T * mu + dfk
        dh[n + 1:n + l+1] = hk + Ae * d
    
    return dh

def beta(ep = None,d = None,mu = None,lam = None,dfk = None,Bk = None,Ae = None,hk = None,Ai = None,gk = None,gamma = None): 
    dh = dah(ep,d,mu,lam,dfk,Bk,Ae,hk,Ai,gk)
    bet = gamma * norm(dh) * min(1,norm(dh))
    return bet

def ddv(ep = None,d = None,lam = None,Ai = None,gk = None): 
    m = np.size(gk)
    dd1 = np.mat(np.zeros((m,m)))
    dd2 = np.mat(np.zeros((m,m)))
    v1 = np.mat(np.zeros((m,1)))
    for i in range(m):
        fm = np.sqrt(lam[i] ** 2 + (gk[i] + Ai[i] * d) ** 2 + 2 * ep ** 2)
        dd1[i,i] = 1 - lam[i] / fm
        dd2[i,i] = 1 - (gk[i] + Ai[i] * d) / fm
        v1[i] = -2 * ep / fm

    return dd1,dd2,v1

def JacobiH(ep = None,d = None,mu = None,lam = None,dfk = None,Bk = None,Ae = None,hk = None,Ai = None,gk = None): 
    n = np.size(dfk)
    l = np.size(hk)
    m = np.size(gk)
    dd1,dd2,v1 = ddv(ep,d,lam,Ai,gk)
    if (l > 0 and m > 0):
        mt1 = np.c_[np.mat([1]), np.mat(np.zeros((1,n))), np.mat(np.zeros((1,l))), np.mat(np.zeros((1,m)))]
        mt2 = np.c_[np.mat(np.zeros((n,1))), Bk, -Ae.T,-Ai.T]
        mt3 = np.c_[np.mat(np.zeros((l,1))), Ae, np.mat(np.zeros((l,l))),np.mat(np.zeros((l,m)))]
        mt4 = np.c_[v1,dd2 * Ai,np.mat(np.zeros((m,l))),dd1]
        A = np.r_[mt1, mt2, mt3, mt4]
       
    if (l == 0):
        mt1 = np.c_[np.mat([1]), np.mat(np.zeros((1,n))), np.mat(np.zeros((1,m)))]
        mt2 = np.c_[np.mat(np.zeros((n,1))), Bk, -Ai.T]
        mt3 = np.c_[v1,dd2 * Ai,dd1]
        A = np.r_[mt1, mt2, mt3]
    
    if (m == 0):
        mt1 = np.c_[np.mat([1]), np.mat(np.zeros((1,n))), np.mat(np.zeros((1,l)))]
        mt2 = np.c_[np.mat(np.zeros((n,1))), Bk, -Ae.T]
        mt3 = np.c_[np.mat(np.zeros((l,1))),Ae,np.mat(np.zeros((l,l)))]
        A = np.r_[mt1, mt2, mt3]
    
    return A

def qpsubp(dfk=None, Bk=None, Ae=None, hk=None, Ai=None, gk=None):
    n = np.size(dfk)
    l = np.size(hk)
    m = np.size(gk)
    gamma = 0.05
    epsilon = 1e-06
    rho = 0.5
    sigma = 0.2
    ep0 = 0.05
    mu0 = 0.05 * np.mat(np.zeros((l,1)))
    lam0 = 0.05 * np.mat(np.zeros((m,1)))   
    d0 = np.mat(np.ones((n,1)))
    u0 = np.r_[np.mat([ep0]), np.mat(np.zeros((n + l + m,1)))]
    z0 = np.r_[np.mat([ep0]), d0, mu0, lam0]
    k = 0
    z = z0
    ep = ep0
    d = d0
    mu = mu0
    lam = lam0
    while (k <= 150):

        dh = dah(ep,d,mu,lam,dfk,Bk,Ae,hk,Ai,gk)
        if (norm(dh) < epsilon):
            break
        A = JacobiH(ep,d,mu,lam,dfk,Bk,Ae,hk,Ai,gk)
        b = beta(ep,d,mu,lam,dfk,Bk,Ae,hk,Ai,gk,gamma) * u0 - dh
        dz = solve(A,b)
        if (l > 0 and m > 0):
            de = dz[0]
            dd = dz[1:n+1]
            du = dz[n + 1:n + l+1]
            dl = dz[n + l + 1:n + l + m+1]
        if (l == 0):
            de = dz[0]
            dd = dz[1:n+1]
            dl = dz[n+1:n + m+1]
        if (m == 0):
            de = dz[0]
            dd = dz[1:n+1]
            du = dz[n+1:n + l+1]
        i = 0
        while (i <= 20):

            if (l > 0 and m > 0):
                dh1 = dah(ep + rho ** i * de,d + rho ** i * dd,mu + rho ** i * du,lam + rho ** i * dl,dfk,Bk,Ae,hk,Ai,gk)
            if (l == 0):
                dh1 = dah(ep + rho ** i * de,d + rho ** i * dd,mu,lam + rho ** i * dl,dfk,Bk,Ae,hk,Ai,gk)
            if (m == 0):
                dh1 = dah(ep + rho ** i * de,d + rho ** i * dd,mu + rho ** i * du,lam,dfk,Bk,Ae,hk,Ai,gk)
            if (norm(dh1) <= (1 - sigma * (1 - gamma * ep0) * rho ** i) * norm(dh)):
                mk = i
                break
            i = i + 1
            if (i == 20):
                mk = 10

        alpha = rho ** mk
        if (l > 0 and m > 0):
            ep = ep + alpha * de
            d = d + alpha * dd
            mu = mu + alpha * du
            lam = lam + alpha * dl
        if (l == 0):
            ep = ep + alpha * de
            d = d + alpha * dd
            lam = lam + alpha * dl
        if (m == 0):
            ep = ep + alpha * de
            d = d + alpha * dd
            mu = mu + alpha * du
        k = k + 1

    val = 0.5 * d.T * Bk * d + dfk.T * d
    return d,mu,lam,val,k

"""
求解二次规划问题
min f(x)=x1^2+2x2^2+x3^2-2x1x2+x3
s.t. x1+x2+x3-4=0
     2x1-x2+x3-2=0
"""
def main1():
    dfk = np.mat([0,0,1]).T
    Bk = np.mat([[2,- 2,0],[- 2,4,0],[0,0,2]])
    Ae = np.mat([[1,1,1],[2,- 1,1]])
    hk = np.mat([- 4,- 2]).T
    Ai = []
    gk = []
    d,mu,lam,val,k = qpsubp(dfk,Bk,Ae,hk,Ai,gk)
    print(d,mu,lam,val,k)


"""
求解二次规划问题
min f(x)=1/2x1^2-x1x2+x2^2-6x1-2x2
s.t. -2x1-x2+3 >= 0
     x1 - x2 + 1 >= 0
     -x1-2x2+2 >= 0
     x1, x2 >= 0
"""
def main2():
    dfk = np.mat([- 6,- 2,- 12]).T
    Bk = np.mat([[2,1,0],[1,4,0],[0,0,0]])
    Ae = np.mat([1,1,1])
    hk = np.mat([- 2]).T
    Ai = np.mat([[1,- 2,0],[1,0,0],[0,1,0],[0,0,1]])
    gk = np.mat([3,0,0,0]).T
    d,mu,lam,val,k = qpsubp(dfk,Bk,Ae,hk,Ai,gk)

    print(d,mu,lam,val,k)

"""
求解二次规划问题
min f(x)=x1^2+x1x2+2x2^2-6x1-2x2-12x3
s.t. x1+x2+x3-2=0
     x1-2x2+3 >= 0
     x1-2x2+3 >= 0
     x1, x2, x3 >= 0
"""
def main3():
    dfk= np.mat([-8,-2]).T
    Bk= np.mat([[1,0],[0,1]])
    # Ae = []
    # hk = []
    Ae = np.mat([])
    hk = np.mat([]).T
    Ai= np.mat([[-2, -4], [ 1.28171817,1], [ 1,0], [ 0,1]]) 
    gk= np.mat([3,2.28171817,4,4]).T

    d,mu,lam,val,k = qpsubp(dfk,Bk,Ae,hk,Ai,gk)

    print(d,mu,lam,val,k)

if __name__ == '__main__':
    main3()
    # main2()
    # main1()