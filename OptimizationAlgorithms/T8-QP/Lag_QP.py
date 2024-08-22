"""
使用Lagrangian方法求解QP问题
"""
import numpy as np
from numpy.linalg import norm, inv, solve

def qlag(H, A, b, c):
    # H, c分别是目标函数的矩阵和向量, A, b 分别是约束条件的矩阵和向量
    # 输出: (x, lam)是KKT点, fval是最优值
    IH = inv(H)
    AHA = A * IH * A.T
    IAHA = inv(AHA)
    AIH = A * IH
    G = IH - AIH.T * IAHA * AIH
    B = IAHA * AIH
    C = -IAHA
    x = B.T*b - G*c
    lam = B*c - C*B
    fval = 0.5 * x.T*H*x + c.T*x
    return x, lam, fval

"""
求解以下QP问题
min x1^2 + 2x2^2 + x3^2 - 2x1x2 + x3
s.t. x1 + x2 + x3 = 4
     2x1 - x2 + x3 = 2
"""
if __name__ == '__main__':
    H = np.mat([[2, -2, 0], [-2, 4, 0], [0, 0, 2]])
    c = np.mat([0, 0, 1]).T
    A = np.mat([[1, 1, 1], [2, -1, 1]])
    b = np.mat([4, 2]).T
    x, lam, fval = qlag(H, A, b, c)
    print(x, lam, fval)