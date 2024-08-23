import numpy as np
from numpy.linalg import norm, pinv

"""
有效集方法
"""

# 求解子问题
def qsubp(H,c,Ae,be): 
    ginvH = pinv(H)
    m = 0
    if np.size(Ae) > 0:
        m = Ae.shape[0]
    if (m > 0):
        rb = Ae * ginvH * c + be
        lambda_ = pinv(Ae * ginvH * Ae.T) * rb
        x = ginvH * (Ae.T * lambda_ - c)
    else:
        x = -ginvH * c
        lambda_ = 0
    return x,lambda_

def qpact(H, c, Ae, be, Ai, bi, x0):
    # x0初始点, H, c是目标函数二次型矩阵和向量
    # Ae = (a1, ..., al)
    # be = (b1, ..., bl)
    # Ai = (al+1, ..., am)
    # bi = (bl+1, ..., bm)
    # 输出 x是最优解, lam是对应的乘子向量
    
    eps = 1e-09
    err = 1e-06
    k = 0
    x = x0
    kmax = 10.0
    ne = np.size(be)
    ni = np.size(bi)
    lamk = np.mat(np.zeros((ne + ni, 1)))
    index = np.mat(np.ones((ni, 1)))

    for i in range(ni):
        if Ai[i] * x > bi[i] + eps:
            index[i] = 0

    while k < kmax:
        Aee = np.mat([])
        if ne>0:
            Aee = Ae
        for j in range(ni):
            if index[j] > 0:
                Aee = np.c_[Aee, Ai[j]]

        Aee.shape = (-1, Ai.shape[1])
        gk = H*x + c
        m1 = 0
        if np.size(Aee) > 0:
            m1 = Aee.shape[0]

        dk, lamk = qsubp(H, gk, Aee, np.mat(np.zeros((m1, 1))))
        if norm(dk) <= err:
            y = 0.0
            if len(lamk) > ne:
                y = np.min(lamk[ne: len(lamk)])
                jk = np.argmin(lamk[ne: len(lamk)])

            if y>=0:
                exitflag = 0
            else:
                exitflag = 1
                for i in range(ni):
                    if index[i] > 0 and (ne+sum(index[0:i]))==jk:
                        index[i] = 0
                        break
            k = k+1
        else:
            exitflag = 1
            alpha = 1.0
            tm = 1.0
            for i in range(ni):
                if index[i] == 0 and Ai[i]*dk < 0:
                    tm1 = ((bi[i] - Ai[i]*x) / (Ai[i]*dk))[0, 0]
                    if tm1 < tm:
                        tm = tm1
                        ti = i
            alpha = min(alpha, tm)
            x = x + alpha * dk
            if tm < 1:
                index[ti] = 1
        if exitflag == 0:
            break

        k = k+1

    fval = 0.5 * x.T * H * x + c.T * x
    return x, lamk, exitflag, fval, k


# 使用上述程序求解问题
# min x1^2-x1x2 + 2x2^2 - x1 - 10x2
# s.t. -3x1 - 2x2 >= -6
#      x1 >= 0, x2 >= 0
def demo1():
    H = np.mat([[2, -1], [-1, 4]])
    c = np.mat([-1, 10]).T
    Ae = np.mat([])
    be = np.mat([])
    Ai = np.mat([[-3, -2], [1, 0], [0, 1]])
    bi = np.mat([-6, 0]).T
    x0 = np.mat([0, 0]).T
    x,lambda_,exitflag,fval,k = qpact(H,c,Ae,be,Ai,bi,x0)

    print(x,lambda_,exitflag,fval,k)

# 求解该问题
# min 1/2x1^2 - x1x2 + x2^2 - 6x1 - 2x2
# s.t. -2x1 - x2 >= -3
#      x1 - x2 >= -1
#      -x1 - 2x2 >= -2
#      x1 >= 0, x2 >= 0
def demo2():
    H = np.mat([[1, -1], [-1, 2]])
    c = np.mat([-6, -2]).T
    Ai = np.mat([[-2, -1], [1, -1], [-1, -2], [1, 0], [0, 1]])
    bi = np.mat([-3, -1, -2, 0, 0]).T
    Ae = np.mat([])
    be = np.mat([])
    x0 = np.mat([0, 0]).T
    x,lambda_,exitflag,fval,k = qpact(H,c,Ae,be,Ai,bi,x0)

    print(x,lambda_,exitflag,fval,k)

if __name__ == '__main__':
    demo2()