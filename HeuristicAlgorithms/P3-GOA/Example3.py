"""
拉压弹簧设计
"""
import numpy as np
import matplotlib.pyplot as plt
import LibGOA

def fun(X):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]

    # 约束条件判断
    g1 = 1-(x2**3*x3) / (71785*x1**4)
    g2 = (4*x2**2-x1*x2) / (12566*(x2*x1**3-x1**4)) + 1/(5108*x1**2)-1
    g3 = 1-(140.45*x1) / (x2**2*x3)
    g4 = (x1+x2) / 1.5-1

    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        fitness = (x3 + 2)*x2*x1**2
    else:
        fitness=1e30
    return fitness

def main():
    pop = 30
    maxIter = 100
    dim = 3
    lb = np.array([0.05, 0.25, 2])
    ub = np.array([2, 1.3, 15])

    fobj = fun
    GbestScore, GbestPosition, Curve = LibGOA.GOA(pop, dim, lb, ub, maxIter, fobj)
    print('Fitness: ', GbestScore)
    print('Optimal Solution: ', GbestPosition)

    plt.figure()
    plt.plot(Curve, 'r-', lw=1)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel('Fitness', fontsize='medium')
    plt.grid()
    plt.title('GOA', fontsize='large')
    plt.show()


if __name__ == '__main__':
    main()