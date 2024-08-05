"""
使用光滑牛顿法求解信赖域子问题

适用于近似Hesse矩阵正定的情况
"""
import numpy as np
from numpy.linalg import norm, inv

def phi(mu, a, b):
    p = a+b-np.sqrt((a-b)**2 + 4 * mu)
    return p


def dah(mu, lam, d, gk, Bk, dta):
    n = len(d)
    dh = np.mat(np.zeros((n+2, 1)))
    dh[0] = mu
    dh[1] = phi(mu, lam, dta**2-norm(d)**2)
    mh=(Bk + lam * np.eye(n)) * d + gk
    mhf = mh.flatten()

    for i in range(n):
        dh[2+i] = mhf[0, i]
    return dh

def beta(mu, lam, d, gk, Bk, dta, gamma):
    dh = dah(mu, lam, d, gk, Bk, dta)
    bet = gamma * norm(dh) * min(1, norm(dh))
    return bet

def jacobiH(mu,lam,d,Bk,dta): 
    n = len(d)
    A = np.mat(np.zeros((n + 2, n + 2)))
    pmu = -4 * mu / np.sqrt((lam + norm(d) ** 2 - dta ** 2) ** 2 + 4 * mu ** 2)
    thetak = (lam + norm(d) ** 2 - dta ** 2) / np.sqrt((lam + norm(d) ** 2 - dta ** 2) ** 2 + 4 * mu ** 2)
    a1 = np.append([1, 0], np.zeros((1,n)))
    a2 = np.append([pmu,1 - thetak],-2 * (1 + thetak) * d.T)
    nz = np.zeros((n,1))
    a3 = Bk + lam * np.eye(n)
    A[0] = np.mat(a1)
    A[1] = np.mat(a2)
    for i in range(n):
        A[2+i] = np.append([0, d[i, 0]], a3[i])

    return A

def trustsq(gk, Bk, dta):
    # 求解信赖域子问题 min qk(d)=gk'*d + 0.5*d'*Bk*d
    # s.t. ||d|| <= delta
    # gk为梯度, Bk是第k次近似Hesse矩阵, dta是半径
    n = len(gk)
    gamma = 0.05; eps=1e-6; rho=0.6
    sigma = 0.2; mu0 = 0.05; lam0 = 0.05
    d0 = np.mat(np.ones((n, 1))) # 初始值为d0
    u0 = np.mat(np.append([mu0], np.zeros((1, n+1)))).T # bar(z)
    z0 = np.mat(np.append([mu0, lam0], d0.T)).T # z0矩阵

    k=0; z=z0; mu=mu0; lam=lam0; d=d0

    while k <= 150:
        dh = dah(mu, lam, d, gk, Bk, dta)
        if norm(dh) < eps:
            break

        A = jacobiH(mu, lam, d, Bk, dta)
        b = beta(mu, lam, d, gk, Bk, dta, gamma)*u0 - dh
        B = inv(A)
        dz = B * b
        dmu = dz[0, 0]
        dlam = dz[1, 0]
        dd = dz[2:n+2, 0]
        m = 0
        mk = 0
        while m < 20:
            dhnew = dah(mu + rho**m*dmu, lam+rho**m*dlam, d+rho**m*dd, gk, Bk, dta)
            if (norm(dhnew) <= ((1 - sigma * (1 - gamma * mu0) * rho ** m) * dh).all()):
                mk = m
                break
            m = m+1

        alpha = rho ** mk
        mu = mu + alpha * dmu
        lam = lam + alpha * dlam
        d = d+alpha*dd
        k = k+1

    val = gk.T*d + 0.5*d.T*Bk*d

    return d, val, lam, k


if __name__ == '__main__':
    gk = np.mat([400, -200]).T # 梯度向量
    Bk = np.mat([[1202, -400], [-400, 200]]) # Hess矩阵
    dta = 5
    d, val, lam, k = trustsq(gk, Bk, dta)
    print(d, val, lam, k) # 信赖域子问题最优解为d=(0, 1), 最优值为qk(d)=-100.