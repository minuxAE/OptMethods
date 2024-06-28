"""
Armijo准则
"""
import numpy as np

def armijo(xk, dk):
    beta = 0.5
    sigma = 0.2
    m = 0
    mmax = 20
    mk = 0
    while m <= mmax:
        if fun(xk + beta ** m * dk) <= fun(xk) + sigma * beta ** m * gfun(xk).T *dk:
            mk = m
            break
        m = m+1

    alpha = beta ** mk
    xk_new = xk + alpha * dk
    fk = fun(xk)
    fk_new = fun(xk_new)

    return mk, alpha, xk_new, fk, fk_new


# 目标函数
def fun(x):
    f = 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    return f

# 梯度函数
def gfun(x):
    gf = np.mat([400 * x[0, 0] * (x[0, 0]**2 - x[1, 0]) + 2* (x[0, 0] - 1),
                 -200*(x[0, 0] ** 2 - x[1, 0])]).T
    return gf

"""
初始迭代点(-1, 1), 下降方向(1, -2)
"""
def main():
    xk = np.mat([-1, 1]).T
    dk = np.mat([1, -2]).T
    _, _, sol, _, fv = armijo(xk, dk)
    print('minimal solution is: ', sol)
    print('value is: ', fv)

if __name__ == '__main__':
    main()