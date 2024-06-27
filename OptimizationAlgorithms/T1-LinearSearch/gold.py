"""
黄金分割法
"""
from math import sqrt, sin
import matplotlib.pyplot as plt

def golds(phi, param, a, b, delta, eps):
    """
    phi: 目标函数
    param: 参数
    a, b: 搜索区间端点
    delta, eps: 自变量和误差
    """
    t = (sqrt(5)-1)/2
    tk, err = [], []
    h = b-a
    phia = eval(phi, globals(), {param: a})
    phib = eval(phi, globals(), {param: b})

    p = a+(1-t)*h
    q = a+t*h

    phip = eval(phi, globals(), {param: p})
    phiq = eval(phi, globals(), {param: q})

    k = 1
    G = [[a, p, q, b]]

    while(abs(phib - phia) > eps or h>delta):
        if phip < phiq: # 计算左试探点
            b = q
            phib = phiq
            q = p
            phiq = phip
            h = b-a
            p = a+(1-t)*h
            phip = eval(phi, globals(), {param: p})
        else: # 计算右试探点
            a = p
            phia = phip
            p = q
            phip = phiq
            h = b-a
            q = a+t*h
            phiq = eval(phi, globals(), {param: q})
        
        err.append(abs(phib-phia))
        tk.append(k)
        k = k+1
        G.append([a, p, q, b]) # n*4的矩阵，存储每次的迭代值

    ds = abs(b-a) # s的误差极限, s为近似极小点
    dphi = abs(phib - phia) # phis的误差极限, phis为近似极小值
    if phip <= phiq:
        s = p
        phis = phip
    else:
        s = q
        phis = phiq

    E = [ds, dphi]
    return s, phis, k, G, E, err, tk

def main():
    # 计算x^2-sin(x)在[0, 1]上的极小点
    delta =1e-4
    eps = 1e-5
    s, phis, k, G, E, err, tk = golds('x**2-sin(x)', 'x', 0, 1, delta, eps)
    print('迭代次数: ', k)
    print('近似极小点: ', s)
    print('近似极小值：', phis)

    # 绘制迭代图像
    plt.plot(tk, err, 'b-.', lw=1)
    plt.xticks(range(0, k, 2))
    plt.yscale('log')
    plt.xlabel('Iteration: k')
    plt.ylabel('Error: err')
    plt.show()


if __name__ == '__main__':
    main()
