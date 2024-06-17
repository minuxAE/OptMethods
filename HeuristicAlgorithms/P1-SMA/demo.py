"""
黏菌算法(SMA)模拟了自然界中黏菌觅食过程中的行为和形态变化
特点: 原理简单, 调节参数少, 寻优能力强, 便于实现

基本算法流程：
S1: 设定参数, 初始化种群, 计算适应度值
S2: 计算重量W和参数a
S3: 生成随机数r, 判断随机数r与参数z的大小, 如果r<z则更新个体位置, 否则更新参数p, vb, vc
S4: 计算适应度值, 更新全局最优解
"""
import numpy as np
# RandValue = np.random.random()

def initialization(pop, ub, lb, dim):
    # 初始化函数
    # pop: 种群数量
    # dim: 每个个体的维度
    # ub: 