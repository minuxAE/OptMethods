"""
蝗虫优化算法, 利用蝗虫群体之间的排斥力和吸引力, 将搜索空间分为排斥空间, 舒适空间和吸引力空间
"""
import numpy as np
from matplotlib import pyplot as plt

# 蝗虫间的社会作用力
def GOA_social_force():
    l = 1.5
    f = 0.5
    d = np.linspace(0, 15, 100)

    s = f*np.exp(-d/l) - np.exp(-d)

    # 绘制适应度函数曲线
    plt.plot(d, s, 'r-', lw=1)
    plt.xlabel('Distance (d)', fontsize='medium')
    plt.ylabel('s(d)', fontsize='medium')
    plt.grid()
    plt.title('l=1.5, f=0.5', fontsize='large')
    plt.show()

def GOA_social_force_m():
    f = np.linspace(0, 1, 5)
    l = 1.5
    d = np.linspace(0, 15, 100)
    for i in range(5):
        s = f[i] * np.exp(-d/l) - np.exp(-d)
        plt.plot(d, s, lw=1, label='f='+str(f[i]))
    
    plt.xlabel('Distance(d)', fontsize='medium')
    plt.ylabel('s(d)', fontsize='medium')
    plt.grid()
    plt.legend()
    plt.title('l=1.5, f=[0, 0.25, 0.5, 0.75, 1]', fontsize='large')
    plt.show()

def GOA_social_force_l():
    l = np.linspace(0, 1, 5)
    f = 0.5
    d = np.linspace(0, 15, 100)

    for i in range(5):
        s = f * np.exp(-d/l[i]) - np.exp(-d)
        plt.plot(d, s, lw=1, label='l='+str(l[i]))
    
    plt.xlabel('Distance(d)', fontsize='medium')
    plt.ylabel('s(d)', fontsize='medium')
    plt.grid()
    plt.legend()
    plt.title('l=[0, 0.5, 1, 1.5, 2], f=0.5', fontsize='large')
    plt.show()


if __name__ == '__main__':
    GOA_social_force_l()