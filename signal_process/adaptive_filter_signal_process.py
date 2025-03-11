'''
该脚本定义一个滤波器对象，该对象具有不同的自适应滤波函数作为其方法
支持
1、LMS算法
2、NLMS算法
3、RLS算法


by kx
'''


import numpy as np



class Adaptive_Filter(object):
    def __init__(self, args):
        self.mu = args.mu
        self.filter_order = args.filter_order
        self.lmbd = args.lmbd
        self.delta = args.delta
        self.N = args.N




    # LMS
    def lms_filter(self, noisy_signal, desired_signal):
        '''
        Parameters
        ----------
        noisy_signal:待滤波信号
        desired_signal：期待信号
        mu：迭代步长
        filter_order：滤波器的阶数
        Returns
        -------
        滤波后的信号
        最优滤波器系数
        '''
        mu = self.mu
        filter_order = self.filter_order
        n_samples = len(noisy_signal)
        weights = np.zeros(filter_order)  # 滤波器初始化
        filtered_signal = np.zeros(n_samples)
        # 迭代滤波（滤波器迭代更新、实时获取滤波器的输出）
        for i in range(filter_order, n_samples):
            x = noisy_signal[i-filter_order:i][::-1]  # 当前时刻的输入翻转
            y = np.dot(weights, x)  # 当前时刻滤波器的输出
            error = desired_signal[i] - y  # 瞬时能量误差
            weights += 2 * mu * error * x  # 权重更新
            filtered_signal[i] = y
        return filtered_signal, weights

    # NLMS
    def nlms_filter(self, x, d):
        mu = self.mu
        N = self.N
        nIters = min(len(x), len(d)) - N
        u = np.zeros(N)
        w = np.zeros(N)
        e = np.zeros(nIters)
        for n in range(nIters):
            u[1:] = u[:-1]
            u[0] = x[n]
            e_n = d[n] - np.dot(u, w)
            w = w + mu * e_n * u / (np.dot(u, u) + 1e-3)  # 正则化
            e[n] = e_n
        return e

    # RLS
    def rls_filter(self, x, d):
        '''
        Parameters
        ----------
        x:自适应滤波器输入信号（噪声源信号）
        d：期望信号（信号源+噪声源）
        N：滤波器阶数
        lmbd：加权因子
        delta：小的正数
        Returns
        -------
        消除回声后的信号
        '''
        N = self.N
        lmbd = self.lmbd
        delta = self.delta
        lmbd_inv = 1 / lmbd
        nIters = min(len(x),len(d)) - N  # 迭代次数=自适应滤波器输出信号的长度
        e = np.zeros(nIters)  # 消除回声后的信号
        w = np.zeros(N)  # 滤波器初始化
        u = np.zeros(N)  # 滤波器输入
        P = np.eye(N)*delta  # 自相关矩阵的逆
        for n in range(nIters):
            u[1:] = u[:-1]  # 序列右移（下一时刻的序列）
            u[0] = x[n]  # 不断更新滤波器输入
            d_p = np.dot(u, w)  # 计算当前时刻滤波器的输出
            e_n = d[n] - d_p  # 当前时刻的误差
            g = np.dot(P, u) / (lmbd + np.dot(u, (np.dot(P, u)))) # Kalman增益向量
            w = w + e_n * g  # 更新权重
            P = lmbd_inv*(P - np.outer(g, np.dot(u, P)))  # 更新自相关向量
            e[n] = e_n
        return e
