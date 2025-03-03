'''
该脚本是一个信号变换及分析函数库，包含各种信号变换和分析方法(主要是时域与频域的变换)：


'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Signal_1d_Processer():
    def __init__(self, args):
        self.args = args
    def add_window(signal):
        window = np.hanning(len(signal))
        signal = signal * window
        return signal

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
    def single_sided_spectrum_transform(signal, sample_rate):
        '''
        该函数用来执行对实数信号的频谱分析，返回具有物理意义的单边谱线和谱线对应的频率刻度
        注意该函数在进行fft变换之前可以选择是否使用归一化处理和加窗处理
        :param signal: 离散信号序列
        :param sample_rate: 信号的采样频率
        :return:
        single_sided_frequencies：谱线对应的频率刻度
        single_sided_fhat ： 谱线幅值
        '''
        n = len(signal)
        mean = np.mean(signal)
        signal = signal - mean  # 零均值处理
        # signal = add_window(signal) * 2  # 加窗处理
        fhat = np.fft.fft(signal, n)  # 计算 FFT
        fhat = fhat / n * 2  # 幅值调整,确保幅值与原信号幅值物理意义相同（因为fft是累计求和所得结果，与信号长度N有关）
        freq = np.fft.fftfreq(len(fhat), d=1 / sample_rate)  # 频率轴

        # 获得单边谱
        half_size = len(fhat) // 2
        single_sided_frequencies = freq[:half_size]
        single_sided_fhat = np.abs(fhat)[:half_size]  # 返回的是fhat的幅值
        title = f'single_sided_spectrum_transform, fs:{sample_rate}'
        return single_sided_frequencies, single_sided_fhat, title

    def single_sided_power_spectrum_transform(signal, sample_rate):
        '''
        该函数用来执行对实数信号的功率谱分析，返回具有物理意义的单边谱线和谱线对应的频率刻度，与时域内信号的能量相对应
        注意该函数在进行fft变换之前可以选择是否使用归一化处理和加窗处理
        :param signal: 离散信号序列
        :param sample_rate: 信号的采样频率
        :return:
        single_sided_frequencies：谱线对应的频率刻度
        single_sided_fhat ： 谱线幅值
        '''
        n = len(signal)
        mean = np.mean(signal)
        signal = signal - mean  # 零均值处理
        fhat = np.fft.fft(signal, n)  # 计算FFT
        # window = np.hanning(n)  # 汉宁窗
        # k = 1.63  # 修正系数保证能量守恒
        # fhat = np.fft.fft(signal * window * k, n)  # 计算FFT
        # freq = np.fft.fftfreq(len(fhat), d=1 / sample_rate)
        freq = np.fft.fftfreq(len(fhat), d=1 / sample_rate)
        PS = fhat * np.conj(fhat) / n  # 功率谱 powerspectrum
        half_size = len(PS) // 2
        single_sided_PSD = np.abs(PS)[:half_size] * 2
        single_sided_freq = freq[:half_size]

        return single_sided_freq, single_sided_PSD

    # 最简单的fft
    def plain_fft_transform(self, signal, sample_rate):
        n = len(signal)
        mean = np.mean(signal)
        signal = signal - mean  # 零均值处理
        # signal = add_window(signal) * 2 # 加窗处理
        fhat = np.fft.fft(signal, n)  # 计算 FFT
        fhat = fhat / n * 2  # 幅值调整,确保幅值与原信号幅值物理意义相同（因为fft是累计求和所得结果，与信号长度N有关）
        freq = np.fft.fftfreq(len(fhat), d=1 / sample_rate)  # 频率轴
        # 获得单边谱
        half_size = len(fhat) // 2
        single_sided_frequencies = freq[:half_size]
        single_sided_fhat = np.abs(fhat)[:half_size]  # 返回的是fhat的幅值
        return single_sided_frequencies, single_sided_fhat

    def average_FFT_spec(self, inputData,  samplingRate, nPerSeg=512 * 2,averaged=True):
        '''
        该函数用来计算信号的平均频谱，输出信号的频谱谱的横坐标及其本身
        注意与一般频谱不同，该函数用来计算平均频谱
        :param inputData: 待分析信号
        :param samplingRate: 采样频率
        :param nPerSeg: 对待分析信号分段后每段的长度
        :param averaged: 是否选择平均的bool值
        :return:
        返回经过平均后的频谱
        '''
        inputData = np.atleast_2d(inputData)  # 将数据转化为二维数组，一个行向量，因为一些运算默认在每一行进行操作
        NyquistFreq = 0.5 * samplingRate  # 信号的奈奎斯特频率，即信号的最高截止频率
        nOverlap = nPerSeg // 2  # 确定两个片段之间重叠的长度

        # This code breaks the input data into a set of segments 将原始信号转换为指定长度片段形成的数组
        step = nPerSeg - nOverlap
        shape = inputData.shape[:-1] + ((inputData.shape[-1]-nOverlap)//step, nPerSeg)  # 新的数据形状
        a = inputData.strides[:-1]
        strides = inputData.strides[:-1] + (step*inputData.strides[-1], inputData.strides[-1])  # 新数据的步幅信息
        inputData = np.lib.stride_tricks.as_strided(inputData, shape=shape, strides=strides)   # 使用新的形状和步幅信息创建一个视图数组

        # Create a hanning window指定使用的窗函数
        window = np.hanning(nPerSeg)  # 创建一个指定长度的窗函数

        # 计算加窗的各段信号的傅里叶变换的平方等效功率谱
        result = inputData - np.expand_dims(np.mean(inputData, axis=-1), axis=-1)  # 0均值化处理
        result = window * result  # 对信号进行加窗
        FFTMag = np.fft.rfft(result, n=nPerSeg)  # 因为输入信号为实信号，使用rfft，返回单边频谱对应的谱值
        FFTMag = np.abs(FFTMag)  #
        if nPerSeg % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2
        # 对傅里叶变换后的幅值进行处理（归一化（考虑序列长度、窗口能量损失））
        scale = 1.0 / (nPerSeg * (window*window).sum())  # 幅值调制（考虑加窗后能量的衰减、归一化）
        FFTMag *= scale
        FFTMag = np.rollaxis(FFTMag, -1, -2)  # 使得一列是一个样本
        # Get the frequencies for the FFT获取频率轴
        FFTFreq = np.fft.rfftfreq(nPerSeg, 1/samplingRate)  # 获取半边频率轴

        if averaged:
            FFTMag = np.mean(FFTMag, axis=-1).flatten()  # 频谱进行平均（一行的元素对应相同的频率成分）
        else:
            FFTMag = FFTMag[0, :, :]

        return FFTFreq.flatten(), FFTMag.real

    def average_power_spec(inputData,samplingRate,nPerSeg=512 * 6,averaged=True):
        ''' 该函数用来计算信号的平均功率谱，输出信号的功率谱的横坐标及其本身
        '''
        # 设置平均傅里叶变换的参数，单个信号片段的长度，相邻两个片段之间重复片段的长度
        inputData = np.atleast_2d(inputData)  # 将数据转化为二维数组，一个行向量，因为一些运算默认在每一行进行操作
        NyquistFreq = 0.5 * samplingRate  # 信号的奈奎斯特频率，即信号的最高截止频率
        nOverlap = nPerSeg // 2  # 确定两个片段之间重叠的长度

        # This code breaks the input data into a set of segments 将原始信号转换为指定长度片段形成的数组
        step = nPerSeg - nOverlap
        shape = inputData.shape[:-1] + ((inputData.shape[-1]-nOverlap)//step, nPerSeg)  # 新的数据形状
        strides = inputData.strides[:-1] + (step*inputData.strides[-1], inputData.strides[-1])  # 新数据的步幅信息
        inputData = np.lib.stride_tricks.as_strided(inputData, shape=shape, strides=strides)

        # Create a hanning window指定使用的窗函数
        window = np.hanning(nPerSeg)  # 创建一个指定长度的窗函数

        # 计算加窗的各段信号的傅里叶变换的平方等效功率谱
        result = inputData - np.expand_dims(np.mean(inputData, axis=-1), axis=-1)  # 0均值化处理
        result = window * result  # 对信号进行加窗
        FFTMag = np.fft.rfft(result, n=nPerSeg)  # 因为输入信号为实信号，使用rfft，返回单边频谱对应的谱值
        FFTMag = np.conjugate(FFTMag) * FFTMag  # 求取复数谱值的模的平方，做功率谱
        if nPerSeg % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2
        # 对傅里叶变换后的幅值进行处理（归一化（考虑序列长度、窗口能量损失））
        scale = 1.0 / (nPerSeg * (window*window).sum())  # 幅值调制（考虑加窗后能量的衰减、归一化）
        FFTMag *= scale
        FFTMag = np.rollaxis(FFTMag, -1, -2)  # 使得一列是一个样本
        # Get the frequencies for the FFT获取频率轴
        FFTFreq = np.fft.rfftfreq(nPerSeg, 1/samplingRate)  # 获取半边频率轴

        if averaged:
            FFTMag = np.mean(FFTMag, axis=-1).flatten()  # 对某个频率成分进行平均（一行的元素对应相同的频率成分）
        else:
            FFTMag = FFTMag[0, :, :]

        return FFTFreq.flatten(), FFTMag.real, result

    # 将加速度信号转换为速度信号
    def acc_to_velsignal(data, sample_rate, filter_freq=10):
        """
          该函数用来根据加速度信号近似求解速度信号，会首先将加速度信号中的低频成分滤去，然后根据速度谱与加速度谱之间的关系求解，返回速度信号
          Parameters
          ----------
          data:1-D list
               振动加速度信号时域序列
          sample_rate: float
                        采样频率
          filter_freq: float
                        截止频率，根据国标选择10hz
          Returns
          -------
          vel_res: 1-D list
                   振动速度信号时域序列
          """
        data = np.array(data)
        n = len(data)
        sos = signal.butter(4, filter_freq, 'highpass', output='sos', fs=sample_rate)  # 高通滤波器
        data = signal.sosfilt(sos, data)  # 这里采用的是sos输出
        fhat = np.fft.fft(data, n)  # 傅里叶变换
        fhat = np.fft.fftshift(fhat)  # 移动到频率中心
        freq = [item for item in
                np.linspace(-sample_rate / 2,
                            sample_rate / 2,
                            n, endpoint=False)]  # 生成频率轴

        jw_list = [2 * np.pi * complex(0, 1) * item for item in freq]  # jw

        vel_jw = []
        for _, (item1, item2) in enumerate(zip(fhat, jw_list)):
            if abs(item2) != 0:
                vel_jw.append(item1 / item2)  # fhat/jw
            else:
                vel_jw.append(complex(0, 0))
        vel_jw = np.array(vel_jw)
        vel_jw = np.fft.ifftshift(vel_jw)  # 从频率中心移出
        vel = np.fft.ifft(vel_jw).real  # 傅里叶逆变换得到时域波形，并取实部
        vel_res = signal.detrend(vel)  # 去除趋势项
        vel_res = vel_res * 9.8 * 1000  # 单位转换为mm
        return vel_res

    def acc_to_velspec(data, sample_rate, filter_freq=10):
        """
            该函数用来根据加速度信号求解速度信号的速度谱横坐标及其本身
            Parameters
            ----------
            data:1-D list
                 振动加速度信号时域序列
            sample_rate: float
                          采样频率
            filter_freq: float
                          截止频率，根据国标选择10hz
            Returns
            -------
            vel_res: 1-D list
                     振动速度信号时域序列
            """
        n = len(data)
        sos = signal.butter(4, filter_freq, 'highpass', output='sos', fs=sample_rate)  # 高通滤波器
        data = signal.sosfilt(sos, data)  # 这里采用的是sos输出
        fhat = np.fft.fft(data, n)  # 傅里叶变换
        fhat = np.fft.fftshift(fhat)  # 移动到频率中心
        freq = [item for item in
                np.linspace(-sample_rate / 2,
                            sample_rate / 2,
                            n, endpoint=False)]  # 生成频率轴

        jw_list = [2 * np.pi * complex(0, 1) * item for item in freq]  # jw

        vel_jw = []
        for _, (item1, item2) in enumerate(zip(fhat, jw_list)):
            if abs(item2) != 0:
                vel_jw.append(item1 / item2)  # fhat/jw
            else:
                vel_jw.append(complex(0, 0))

        vel_jw = np.fft.ifftshift(vel_jw)  # 从频率中心移出
        vel = np.fft.ifft(vel_jw).real  # 傅里叶逆变换得到时域波形，并取实部
        vel_res = signal.detrend(vel)  # 去除趋势项
        vel_res = vel_res * 9.8 * 1000  # 单位转换为mm
        fftFreqs, fftAmps, rawSignal = generateFFT_spec(vel_res, samplingRate=sample_rate, averaged=True)
        return fftFreqs, fftAmps


if __name__ == '__main__':
    # 测试信号加载
    # 信号1：正弦合成信号
    sample_rate = 1000  # 采样率
    duration = 2.0  # 采样时间
    t = np.arange(0, duration, 1 / sample_rate)
    signal0 = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # 两个不同频率信号相加
    signal = signal0 + 2.5 * np.random.randn(len(t))  # 添加噪声

    # 信号2：轴承内圈故障模拟信号
    # fs = 12e3  # 采样频率12k
    # fr = 30  # 转频,转1圈所需时间为0.03s
    # fn = 4e3  # 轴承固有频率4k
    # C = 700  # 固有振动衰减稀疏衰减系数
    # A0 = 0.3  # 振动幅度常数
    # Nb = 4  # 滚珠个数
    # T = 1 / (4 * fr)  # 滚珠通过周期
    # N = 2000  # 构造信号的长度
    # SNR = -13  # 信噪比
    # NT = round(fs * T)  # 一个冲击周期的长度
    # tt0 = np.arange(0, NT / fs, 1 / fs)  # 一个冲击周期内的采样时刻
    # tt = np.arange(0, N / fs, 1 / fs)  # 采样期间内的采样时刻
    # p1 = int(np.ceil(N / NT) - 1)  # 采样期间冲击重复的次数
    # s = []  # 用来存储信号数组
    # for i in range(p1):
    #     tt1 = np.arange((i * NT) / fs, ((i + 1) * NT) / fs, 1 / fs)  # 第i个周期内的采样时刻
    #     s.append((1 + A0 * np.cos(2 * np.pi * fr * tt1)) * np.exp(-C * tt0) * np.cos(2 * np.pi * fn * tt0))
    # s = np.concatenate(s)
    # d = N - len(s)  # p1次周期后剩下的采样点数
    # ttt0 = np.arange(0, d / fs, 1 / fs)  # p1次周期后一个周期内剩下的采样时刻
    # ttt1 = np.arange(p1 * NT / fs, N / fs, 1 / fs)  # p1次周期后对应的采样时刻
    # s_r = np.array((1 + A0 * np.cos(2 * np.pi * fr * ttt1)) * np.exp(-C * ttt0) * np.cos(2 * np.pi * fn * ttt0))
    # s = np.concatenate((s, s_r))
    # signal_power = 1 / len(s) * np.sum(s ** 2)  # 计算信号的功率，用于添加固定信噪比的噪声
    # snr_linear = 10 ** (SNR / 10)  # 计算信噪比所需的噪音功率
    # noise_power = signal_power / snr_linear  # 噪声的功率
    # noise_stddev = np.sqrt(noise_power)  # 计算噪音的标准差（方差的平方根）
    # white_noise = np.random.normal(0, noise_stddev, len(s))  # 生成白噪声信号
    # s_n = s + white_noise

    # 测试
    freq, fhat = average_FFT_spec(signal, sample_rate, nPerSeg=512 * 2, averaged=True)
    # freq, fhat = single_sided_spectrum_transform(signal, sample_rate)
    # freq1, fhat1 = single_sided_spectrum_transform(signal0, sample_rate)

    # 可视化
    plt.figure(1)
    plt.plot(freq, fhat, color='c', linewidth=1.5, label='Freq')
    # plt.plot(freq1, fhat1, color='b', linewidth=1.5, label='Freq')
    plt.xlim(freq[0], freq[-1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    print(1)