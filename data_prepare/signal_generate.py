'''
该脚本用来实现对几种不同类型信号的仿真生成，有如下类型
合成正弦信号
chrip信号
滚动轴承内圈故障信号
'''
import numpy as np
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D



# 生成一个模拟声音信号，该信号由发生在不同时刻的一段衰减的不同频率的正弦信号组合而成
def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    T : 指定构造信号的长度
    num_intervals :决定信号分量的个数
    gamma:能量衰减因子

    """
    rng = np.random.RandomState(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(1):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window

    return x




# 生成一个模拟内圈故障的信号
def analog_I_F_signal_generate(fs, fr, fn, C, A0, Nb, N, SNR):
    '''

    Parameters
    ----------
    fs : 采样频率
    fr ： 转频
    fn ： 共振频率
    C ： 阻尼衰减系数
    A0 ： 振动幅值
    Nb ：轴承滚珠个数
    N ： 构造信号的长度
    SNR ： 信噪比
    Returns
    -------
    s : 不含噪声的模拟信号
    s_n : 含噪声的模拟信号
    '''
    T = 1 / (4 * fr)  # 滚珠通过周期
    NT = round(fs * T)  # 一个冲击周期的长度
    tt0 = np.arange(0, NT / fs, 1 / fs)  # 一个冲击周期内的采样时刻
    tt = np.arange(0, N / fs, 1 / fs)  # 采样期间内的采样时刻
    p1 = int(np.ceil(N / NT) - 1)  # 采样期间冲击重复的次数
    s = []  # 用来存储信号数组
    # 构造信号
    for i in range(p1):
        tt1 = np.arange((i * NT) / fs, ((i + 1) * NT) / fs, 1 / fs)  # 第i个周期内的采样时刻
        tt1 = tt1[0:NT]
        s.append((1 + A0 * np.cos(2 * np.pi * fr * tt1)) * np.exp(-C * tt0) * np.cos(2 * np.pi * fn * tt0))
    s = np.concatenate(s)
    d = N - len(s)  # p1次周期后剩下的采样点数
    ttt0 = np.arange(0, d / fs, 1 / fs)  # p1次周期后一个周期内剩下的采样时刻
    ttt1 = np.arange(p1 * NT / fs, N / fs, 1 / fs)  # p1次周期后对应的采样时刻
    s_r = np.array((1 + A0 * np.cos(2 * np.pi * fr * ttt1)) * np.exp(-C * ttt0) * np.cos(2 * np.pi * fn * ttt0))
    s = np.concatenate((s, s_r))
    # s[1: NT] = 0;
    # 信号添加噪声
    signal_power = 1 / len(s) * np.sum(s ** 2)  # 计算信号的功率
    # 计算信噪比所需的噪音功率
    snr_linear = 10 ** (SNR / 10)
    noise_power = signal_power / snr_linear  # 噪声的功率
    noise_stddev = np.sqrt(noise_power)  # 计算噪音的标准差（方差的平方根）
    white_noise = np.random.normal(0, noise_stddev, len(s))  # 生成白噪声信号
    s_n = s + white_noise
    return s, s_n


# 生成一个高斯信号
def gaussian(x, x0, sigma):
    '''
    该函数用来产生一个指定参数的高斯函数（钟形函数）
    :param x: 采样时刻序列
    :param x0: 极值点的时刻
    :param sigma: 调整钟形函数的宽窄
    :return: 返回一个钟形函数
    '''
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

def make_chirp(t, t0, a):
    '''
    该函数用来生成二次chrip信号
    :param t: 采样时刻序列
    :param t0: 时间偏移量，影响频率的计算，影响chrip信号在不同时刻对应的频率，调整频率曲线的起始点，起始频率的大小
    :param a: 速度参数，决定了信号频率变换的快慢
    :return:
    '''
    frequency = (a * (t + t0)) ** 2
    chirp = np.sin(2 * np.pi * frequency * t)
    return chirp, frequency


if __name__ == '__main__':
    # region测试信号加载
    # 信号1：正弦合成信号
    # sample_rate = 1000  # 采样率
    # duration = 2.0  # 采样时间
    # t = np.arange(0, duration, 1 / sample_rate)
    # signal0 = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # 两个不同频率信号相加
    # signal = signal0 + 2.5 * np.random.randn(len(t))  # 添加噪声
    #
    # # 信号2：轴承内圈故障仿真信号
    # fs = 12e3  # 采样频率12k
    # fr = 30  # 转频,转1圈所需时间为0.03s
    # fn = [2e3, 3e3, 4e3, 5e3]  # 轴承固有频率4k
    # C = 1000  # 固有振动衰减稀疏衰减系数
    # A0 = 0.3  # 振动幅度常数
    # Nb = 4  # 滚珠个数
    # N = 2000  # 构造信号的长度
    # SNR = 3  # 信噪比
    # for i in range(4):
    #     fn1 = fn[i]
    #     s, s_n = analog_I_F_signal_generate(fs=fs, fr=fr, fn=fn1, C=C, A0=A0, Nb=Nb, N=N, SNR=SNR)
    #     time = np.arange(len(s_n)) / fs
    #     # 不同形式的分析数据
    #     f, t, Zxx = signal.stft(s_n, fs, nperseg=256, noverlap=250)  #
    #     # f, t, Zxx = signal.stft(desired_data, fs, nperseg=2048)  #
    #     Zxx = np.abs(Zxx)
    #     # 信号预处理
    #     e_value = np.sqrt(sum(s_n ** 2) / len(s_n))
    #     print('{}有效值:{}'.format(fn1, e_value))  # 信号的均方值
    #     # 采用不同超参数的算法
    #     # 原始信号
    #     plt.title('原始信号')
    #     plt.plot(time, s)
    #     plt.xlabel('时间')
    #     plt.ylabel('幅值')
    #     plt.show()
    #     # 结果可视化
    #     plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    #     plt.title('I/30Hz/12e3/256/250/{}'.format(fn1))
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.show()
    #
    #     # 信号3：chrip信号
    #     time = np.linspace(0, 1, 2000)  # 设置时间刻度
    #     chirp1, frequency1 = make_chirp(time, 0.2, 9)
    #     chirp2, frequency2 = make_chirp(time, 0.1, 5)
    #     chirp = chirp1 + 0.6 * chirp2
    #     chirp *= gaussian(time, 0.5, 0.2)
    # endregion


    # 模拟信号测试
    T = 2 ** 13  # 8192
    x = generate_harmonic_signal(T)
    plt.figure(figsize=(8, 2))
    plt.plot(x)
    plt.title("Original signal")
    plt.figure(figsize=(8, 4))
    plt.specgram(x, Fs=1024)
    plt.title("Time-Frequency spectrogram of signal")
    # 散射分析
    J = 6
    Q = 16
    scattering = Scattering1D(J, T, Q)
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    Sx = scattering(x)

    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(Sx[order0][0])
    plt.title('Zeroth-order scattering')
    plt.subplot(3, 1, 2)
    plt.imshow(Sx[order1], aspect='auto')
    plt.title('First-order scattering')
    plt.subplot(3, 1, 3)
    plt.imshow(Sx[order2], aspect='auto')
    plt.title('Second-order scattering')
    plt.tight_layout()
    plt.show()