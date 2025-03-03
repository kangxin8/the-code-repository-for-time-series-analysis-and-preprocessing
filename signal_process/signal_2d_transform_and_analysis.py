'''
该脚本是一个信号变换及分析函数库，包含各种信号变换和分析方法(主要是时域与时频域的变换)：
函数特点：

'''
import pywt
from matplotlib import pyplot as plt
from scipy import signal
import torch
import scipy.io.wavfile
from data_prepare.utils import *
from kymatio.torch import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from utils.plot_utils import visuallize_2d_array
from kymatio.scattering1d.frontend.base_frontend import ScatteringBase1D
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from io import BytesIO
from PIL import Image
from skimage.transform import resize

class ModulusStable(Function):
    """Stable complex modulus

    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning nans in all cases.

    Usage
    -----
    modulus = ModulusStable.apply  # apply inherited from Function
    x_mod = modulus(x)

    Parameters
    ---------
    x : tensor
        The complex tensor (i.e., whose last dimension is two) whose modulus
        we want to compute.

    Returns
    -------
    output : tensor
        A tensor of same size as the input tensor, except for the last
        dimension, which is removed. This tensor is differentiable with respect
        to the input in a stable fashion (so gradent of the modulus at zero is
        zero).
    """
    @staticmethod
    def forward(ctx, x):
        """Forward pass of the modulus.

        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        x : tensor
            The complex tensor whose modulus is to be computed.

        Returns
        -------
        output : tensor
            This contains the modulus computed along the last axis, with that
            axis removed.
        """
        ctx.p = 2
        ctx.dim = -1
        ctx.keepdim = False

        output = (x[...,0] * x[...,0] + x[...,1] * x[...,1]).sqrt()

        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the modulus

        This is a static method which does not require an instantiation of the
        class.

        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        grad_output : tensor
            The gradient with respect to the output tensor computed at the
            forward pass.

        Returns
        -------
        grad_input : tensor
            The gradient with respect to the input.
        """
        x, output = ctx.saved_tensors

        if ctx.dim is not None and ctx.keepdim is False and x.dim() != 1:
            grad_output = grad_output.unsqueeze(ctx.dim)
            output = output.unsqueeze(ctx.dim)

        grad_input = x.mul(grad_output).div(output)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input
def cdgmm(A, B):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2).
        B : tensor滤波器组
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).
        inplace : boolean, optional
            If set to True, all the operations are performed in place.

        Raises
        ------
        RuntimeError
            In the event that the filter B is not a 3-tensor with a last
            dimension of size 1 or 2, or A and B are not compatible for
            multiplication.

        TypeError
            In the event that A is not complex, or B does not have a final
            dimension of 1 or 2, or A and B are not of the same dtype, or if
            A and B are not on the same device.

        Returns
        -------
        C : tensor
            Output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].

    """
    # Ensure A and B are compatible for multiplication
    assert A.shape[-1] == 2, "A should have a last dimension of size 2 (real and imaginary parts)."
    assert B.shape[-1] in [1, 2], "B should have a last dimension of size 1 (real) or 2 (complex)."

    # If B is real, we append a zero imaginary part
    if B.shape[-1] == 1:
        return A * B
        # B = torch.cat([B, torch.zeros_like(B)], dim=-1)  # (M, N, 2)
    else:
        # Split real and imaginary parts of A and B
        A_real, A_imag = A[..., 0], A[..., 1]
        B_real, B_imag = B[..., 0], B[..., 1]

        # Perform complex multiplication
        C_real = A_real * B_real - A_imag * B_imag
        C_imag = A_real * B_imag + A_imag * B_real

        # Combine real and imaginary parts
        C = torch.stack([C_real, C_imag], dim=-1)

        return C

# stft变换
def stft_window(sig, fs, nperseg=256, noverlap=250):
    '''
    该函数用来使用指定的超参数对信号样本进行stft变换 ，单次fft变换的长度，相邻两次变换样本点重叠的个数
    Parameters
    ----------
    sig：待分析的样本信号
    fs：信号的采样频率
    nperseg：单次fft变换截取信号的长度，注意该参数由于要显示重复瞬态冲击，因此不能太长；另一方面如果太短将会丢失该长度及以上对应的频率成分
    noverlap：相邻两次变换重叠的点数
    Returns：返回该样本对应的值，以及横纵坐标索引
    -------
    f:返回的频率成分刻度
    t：返回时间轴刻度
    Y：返回时频幅值
    title：返回改图的信息
    '''
    # print(sig.shape)
    w_len = nperseg
    overlap = noverlap
    f, t, amp = signal.stft(sig, fs, nperseg=w_len, noverlap=overlap)
    amp = np.abs(amp)
    amp = np.array(amp)
    title = f'STFT,Sig_length:{len(sig)}, fs:{fs}, Win_length:{nperseg}, Overlap:{noverlap}'
    return f, t, amp, title

# 小波变换CWT
class WaveletTransform:
    '''
    该对象相当于一个针对某种信号的分析器，实例化该对象需要指定小波变换的类型、和待分析信号的采样频率
    '''
    def __init__(self, wavelettype='cmor', sample_rate=1, rot_freq=25):
        self.wavelet = wavelettype
        self.sample_rate = sample_rate
        self.rot_freq = rot_freq
    # 连续小波变换
    def cwt(self, data, scales):
        '''
        该函数用来实现对待分析信号的离散小波变换
        :param data: 待分析信号
        :param scales: 期望采用的尺度序列
        :return:
        连续小波变换到数组，一行代表一个尺度，第一行对应尺度序列中的第一个尺度
        每一个尺度对应的小波函数的中心频率，单位hz
        每个成分对应的时刻
        此次分析的基本信息
        '''
        sampling_period = 1 / self.sample_rate  # 计算采样间隔,单位s
        duration = (len(data)-1) * sampling_period  # 样本采样时长
        coef, freqs = pywt.cwt(data, scales, self.wavelet, sampling_period)  # 调用cwt函数,尺度a越大，越越对应低频成分
        time = np.linspace(0, duration, len(data))  # 生成一个时间轴
        title = f'cwt, {self.wavelet}, scale_number:{len(scales)}, fs:{self.sample_rate}'
        return coef, freqs, time, title

    # 平稳小波变换（不抽样的离散小波变换）
    def swt(self, data, level=None, start_level=0, trim_approx=False, norm=False):
        '''
        该函数用来执行平稳离散小波变换
        :param data: 待分析数据
        :param level: 指定分解等级
        :param start_level: 可以选择跳过一些等级
        :param trim_approx: bool值，如果是返回的系数的组织方式与wavedec相同；如果不是，就返回每一等级的一对系数（近似和细节）
        :param norm: bool值，如果是，将所有系数拼接后形成的序列的能量值与原始信号能量值相同
        :return:
        '''
        sampling_period = 1 / self.sample_rate  # 计算采样间隔,单位s
        duration = (len(data) - 1) * sampling_period  # 样本采样时长
        level = int(np.ceil(np.log2(self.sample_rate/self.rot_freq)))  # 计算分辨转频所需的最小分解等级
        coef = pywt.swt(data, self.wavelet, level=level, start_level=0, axis=-1, trim_approx=True, norm=True)
        coef = np.array(coef)  # 系数列表形式转换为2d数组形式
        coef = np.flipud(coef)  # 表示反转2d数组顺序，使得，第一行对应最高频成分，最后一行对应最低频成分
        freqs = [(2**i)*np.floor((self.sample_rate/(2**(level+1)))) for i in range(level+1)]  # 生成频率轴,注意是每个频带右侧边界对应的频率
        freqs = np.flipud(freqs)
        time = np.linspace(0, duration, len(data))  # 生成一个时间轴
        title = f'swt, {self.wavelet}, level:{level}, fs:{self.sample_rate}'
        return coef, freqs, time, title

    # 单级离散小波变换
    def dwt(self, data, mode='symmetric'):
        '''
        该函数用来执行对信号的单级离散小波变换，返回近似系数（低频成分），和高频细节系数
        :param data: 待分析信号
        :param mode: 信号扩展的方式，将决定输出系数的长度
        :return: 低频近似系数和高频细节系数，长度与信号扩展方式有关
        '''
        (cA, cD) = pywt.dwt(data, self.wavelet, mode=mode)
        return cA, cD
    # 多级离散小波变换
    def wavedec(self, data, mode='symmetric', level=None):
        '''
        该函数用来实现指定分解等级的离散小波变换，使用mallat算法（多分辨分析），分解后的系数不适于做可视化，逐级系数递减，只能用于提取指标
        :param data: 待分析数据
        :param mode: 信号延拓方式
        :param level: 分解等级，如果不指定将默认使用最大分解等级
        :return:
        返回信号在指定分解等级处的近似系数和细节系数形成的列表，每个列表的元素都是数组
        '''
        coefficients_list = pywt.wavedec(data, self.wavelet, mode=mode, level=level)
        return coefficients_list
    # 只计算近似系数（低频成分）或者细节系数（高频成分）
    def downcoef(self, part, data, mode='symmetric', level=1):
        '''
        该函数适用于只需要某个等级的近似系数或者细节系数时使用，注意使用时对信号的长度有严格的要求，需要控制信号的长度为2的n次幂
        :param part:；’a‘,'d'两种可选，前者表示近似系数，后者表示细节系数
        :param data:待分析数据
        :param mode:信号延拓方式
        :param level:分解等级，默认为1
        :return:
        返回指定等级的近似系数或者细节系数
        '''
        coffes = pywt.downcoef(part, data, self.wavelet, mode=mode, level=level)


        return coffes

    # 小波包变换
    def wp_coefficients(self, signal, d_level=None):
        """
        该函数用来计算指定信号的小波包频带分解系数，返回每一层小波包变换的系数对应的时域坐标以及系数值
        :param signal: 待分析信号
        :param d_level:  指定小波包分解层次
        :param sample_rate: 信号的采样频率
        :return: 返回每一层小波包变换的系数对应的时域坐标以及系数值
        """
        sampling_period = 1 / self.sample_rate  # 计算采样间隔,单位s
        if d_level is None:
            d_level = int(np.fix(np.log2(self.sample_rate / self.rot_freq)))  # 小波包分解层数
        else:
            d_level = int(d_level)
        wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=d_level)
        map = {}
        t_map = {}
        map[1] = signal
        t_map[1] = np.linspace(0, (len(map[1]) - 1) * sampling_period, len(map[1]))
        for row in range(1, d_level + 1):  # 是从一开始不是从0开始
            for i in [node.path for node in wp.get_level(row, 'freq')]:  # 频率从低到高
                map[i] = wp[i].data
                t_map[i] = np.linspace(0, (len(map[1]) - 1) * sampling_period, len(map[i]))
        return t_map, map


    # 尺度到实际频率的变换
    def scale_to_frequency(self, scale, precision=10):
        '''
        该函数用来实现，计算对于指定类型的小波函数在某个尺度下对应的实际频率
        :param scale: 尺度因子的值，标量
        :param precision: 精度，决定了离散化小波时的采样点数
        :return:
        对应采样频率下的实际频率
        '''
        norm_freq = pywt.scale2frequency(self.wavelet, scale, precision)  # 相对于采样频率的归一化频率
        freq = norm_freq * self.sample_rate  # 实际频率
        return freq

    # 实际频率到尺度的变换
    def frequency_to_scales(self, freq, presion=10):
        '''
        该函数用来实现对于指定类型的小波函数在某个期望分析频率下对应的尺度
        :param freq:  期望分析的频率
        :param presion: 精度，决定了离散化小波时的采样点数
        :return: 返回对应的尺度
        '''
        norm_freq = freq / self.sample_rate
        scales = pywt.frequency2scale(self.wavelet, norm_freq, precision=10)
        return scales
    # 查看指定小波类型的基本信息
    def wavelet_info(self):
        '''
        该函数用来返回使用的小波类型的信息
        滤波器的长度、正交、双正交、对称等属性，以及是否可用作cwt或者dwt
        :return:
        '''
        wavelet = pywt.Wavelet(self.wavelet)
        return wavelet

    # 返回小波函数离散序列


    # 小波函数可视化
    # 静态方法：获取pywt库的各种信息
    @staticmethod
    def info_of_pywt():
        '''该函数用来查看各种小波类型的信息'''
        wavelist = pywt.wavelist(kind='continuous')  # 返回所有cwt适用的小波类型

class STFT_TF_analysis(object):
    def __init__(self, fs=12e3, nperseg=256, noverlap=250, non_threshod=None):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.non_threshod = non_threshod

    # 获取stft结果数组
    def stft_results_obtain(self, x):
        w_len = self.nperseg
        overlap = self.noverlap
        f, t, amp = signal.stft(x, self.fs, nperseg=w_len, noverlap=overlap)
        amp = np.abs(amp)
        return f, t, amp

    # 获取stft+非线性处理的数组
    def stft_non_results_obtain(self, x):
        w_len = self.nperseg
        overlap = self.noverlap
        fr, t, amp = signal.stft(x, self.fs, nperseg=w_len, noverlap=overlap)
        amp = np.abs(amp)
        # 非线性处理
        quartiles = np.percentile(amp.flatten(), [25, 50, self.non_threshod])  # 四分位数
        origin_yvalue = 0.001  # 超参数1的选择
        half_xvalue = quartiles[2] * 100.0  # 超参数2的选择
        f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        ZXX_non = amp * 100.0  # 标准化时频图,非线性函数的输入
        stft_ZXX_non = f(ZXX_non)  # 非线性处理
        stft_ZXX_non_resize = resize(stft_ZXX_non.numpy(), output_shape=(224, 224), anti_aliasing=True)  # 尺寸校准
        return stft_ZXX_non_resize
    # 获取stft+非线性获取的时频图
    def stft_non_tf_results_obtain(self, x):
        w_len = self.nperseg
        overlap = self.noverlap
        fr, t, amp = signal.stft(x, self.fs, nperseg=w_len, noverlap=overlap)
        amp = np.abs(amp)
        # 非线性处理
        quartiles = np.percentile(amp.flatten(), [25, 50, self.non_threshod])  # 四分位数
        origin_yvalue = 0.001  # 超参数1的选择
        half_xvalue = quartiles[2] * 100.0  # 超参数2的选择
        f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        ZXX_non = amp * 100.0  # 标准化时频图,非线性函数的输入
        stft_ZXX_non = f(ZXX_non)  # 非线性处理

        return fr, t, stft_ZXX_non

    # 获取时频图像
    def TFimage_obtain(self, x):
        w_len = self.nperseg
        overlap = self.noverlap
        f, t, amp = signal.stft(x, self.fs, nperseg=w_len, noverlap=overlap)
        amp = np.array(amp)
        fig1, ax1 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm = ax1.pcolormesh(t, f, np.abs(amp), cmap='viridis', norm=None, vmin=None,
                             vmax=None, shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ax1.axis('off')  # 移除坐标轴
        plt.tight_layout(pad=0)  # 调整图形以移除白边
        buf_original = BytesIO()  # 将图形保存到内存中的BytesIO对象
        plt.savefig(buf_original, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)  # 关闭图形以释放内存
        buf_original.seek(0)  # 将BytesIO对象转换为Image对象
        original_img = Image.open(buf_original)
        # plt.imshow(original_img)
        # plt.axis('off')  # 去掉坐标轴
        # plt.show()
        return original_img

# 小波散射源码执行
class wavelet_scat_embeded_block(nn.Module, ScatteringBase1D):
    '''
    该类称为小波散射模块，模块的输入是原始的一批次振动信号，然后执行一阶小波散射操作，输出一个批次的每个信号的一阶小波散射表示
    '''
    def __init__(self, J=10, shape=2048, Q=11, T=None, max_order=2, average=None,
                 oversampling=0, out_type='array', backend='torch'):
        nn.Module.__init__(self)
        self.frontend_name = 'torch'
        ScatteringBase1D.__init__(self, J, shape, Q, T, max_order, average,
                oversampling, out_type, backend)  # 初始化小波散射对象（参数应包含J,Q,x）
        ScatteringBase1D.build(self)  # 计算填充后信号的长度、左填充和右填充的长度以及索引共5个超参数
        ScatteringBase1D.create_filters(self)  # 根据超参数创建滤波器组
        self.register_filters()  # 将滤波器参数计入buffer
        # 滤波器组(n_filters, filter_length, 1)
        buffer_dict = dict(self.named_buffers())  # 获取滤波器组，字典形式
        self.psi1_matrix = torch.stack([buffer_dict['tensor' + str(n)] for n in range(len(buffer_dict))],
                                  dim=0)  # 将滤波器堆叠为一个高维矩阵（size,）
    def register_filters(self):
        """ This function run the filterbank function that
        will create the filters as numpy array, and then, it
        saves those arrays as module's buffers."""
        n = 0
        # prepare for pytorch
        # for level in range(len(self.phi_f['levels'])):
            # self.phi_f['levels'][level] = torch.from_numpy(
            #     self.phi_f['levels'][level]).float().view(-1, 1)
            # self.register_buffer('tensor' + str(n), self.phi_f['levels'][level])
            # n += 1
        for psi_f in self.psi1_f:
            for level in range(len(psi_f['levels'])):
                psi_f['levels'][level] = torch.from_numpy(
                    psi_f['levels'][level]).float().view(-1, 1)
                self.register_buffer('tensor' + str(n), psi_f['levels'][level])
                n += 1
        # for psi_f in self.psi2_f:
        #     for level in range(len(psi_f['levels'])):
        #         psi_f['levels'][level] = torch.from_numpy(
        #             psi_f['levels'][level]).float().view(-1, 1)
        #         self.register_buffer('tensor' + str(n), psi_f['levels'][level])
        #         n += 1

    def forward(self, x):  # x->(batch, 1, n_samples)
        '''
        输入是一个批次的原始振动信号
        Parameters
        ----------
        x：`torch.Tensor` (batch_size, 1, n_samples)

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of scat filters activations.
        '''
        # 输入形状的确定
        signal_shape = x.shape[-1:]
        x = x.reshape((-1, 1) + signal_shape)  # 形如（batch,1,signal_size）的形状
        # 滤波操作
        # 将原始信号填充，并进行傅里叶变换(batch_size, 1, signal_size, 2)
        U_0 = F.pad(x, (self.pad_left, self.pad_right), mode='reflect')  # 对原始信号进行扩展
        U_0 = U_0[..., None]  # 扩展一个新的维度（batch,1,signal_size,1）
        x_r = torch.zeros(U_0.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = U_0[..., 0]
        U_0_hat = torch.view_as_real(torch.fft.fft(torch.view_as_complex(x_r)))  # 将复数形式的虚部和实部作为两个实数处理
        # 与滤波器执行频域的乘积
        U_1_c = cdgmm(U_0_hat, self.psi1_matrix)  # 滤波结果（batch, n_filters, signal_size, 2）
        # 对结果进行采样(不采样)
        U_1_c = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(U_1_c)))  # 逆傅里叶变换
        U_1_m = ModulusStable.apply(U_1_c)  # 取模
        first_scat_features = U_1_m[..., self.ind_start[0]:self.ind_end[0]]  # 对结果进行裁剪，消除填充的影响(batch_size, n_filters, signal_size)
        first_scat_features = first_scat_features.squeeze(0).squeeze(0).numpy()
        # 正则化处理
        # mean = first_scat_features.mean()
        # std = first_scat_features.std()
        # first_scat_features = (first_scat_features - mean) / std
        # max = first_scat_features.max() * 0.5
        # min = first_scat_features.min()
        # first_scat_features = (first_scat_features - min) / (max - min)
        # 测试滤波结果（可视化）
        # time_duration = (first_scat_features.shape[2] - 1) / fs
        # t = np.linspace(0, time_duration, first_scat_features.shape[2])  # 时间轴信息
        # freq_order1 = [self.psi1_f[n]['xi'] * fs for n in range(len(self.psi1_f))]  # 频率轴信息
        # numpy_array = first_scat_features.squeeze(0).numpy()
        # visuallize_2d_array(numpy_array, X=t, Y=freq_order1)
        return first_scat_features  # (batch_size, n_filters, signal_size)


# 小波散射
class wavelet_scaterring_analysis:
    '''
    该对象用来建立一个小波散射对象，可执行各种散射操作
    J表示最大尺度数
    Q表示单倍频程包含小波个数
    '''
    def __init__(self, J, Q):
        self.J = J
        self.Q = Q

    # 标准小波变换
    def standard_scattering(self, signal):
        T = signal.shape[-1]
        scattering = Scattering1D(self.J, T, self.Q, out_type='list')
        scattering_coefficients = scattering(signal)
        meta = scattering.meta()
        # 一阶散射系数不同频带对应的中心频率
        # 二阶散射系数不同频带对应的中心频率
        return scattering_coefficients, meta
    # 返回不进行低通滤波的散射系数
    def no_lowpass_scattering(self, signal):
        T = 0
        scattering = Scattering1D(self.J, T, self.Q, out_type='list')
        scattering_coefficients = scattering(signal)
        meta = scattering.meta()
        # 一阶散射系数不同频带对应的中心频率
        # 二阶散射系数不同频带对应的中心频率
        return scattering_coefficients, meta
    # 1级小波散射

    # 提升采样频率

    # 返回小波函数
    def generate_filters(self, N):
        '''
        该函数用来生成执行小波散射的滤波器组中的每个小波函数的离散化形式
        :param signal: 输入信号
        :return: 以字典形式返回离散化小波函数
        低通滤波器：包含子采样级别的低通滤波器形式，相邻两个级别低通滤波器样本点数相差2倍
        1级小波散射滤波器组小波函数
        2级小波散射滤波器组小波函数
        参数中两个N的含义：第一个表示填充后的分析信号的长度；第二个表示低通滤波器的时域支撑范围
        '''
        # N = signal.shape[-1]  # 信号的长度
        phi_f, psi1_f, psi2_f = scattering_filter_factory(N, self.J, self.Q, N)
        return phi_f, psi1_f, psi2_f


    # 该方法只返回一阶散射结果而不进行可视化
    def scattering_result(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        time_duration = (T - 1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引

        # 使用重复采样方法将一阶散射的结果转换为可进行可视化的2d数组
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]  # 将不同子带内的小波系数扩展为相同的长度便于可视化
        coefficients_order1 = np.array(coefficients_order1)
        # # 对散射系数进行正则化处理
        # sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]  # 一阶散射滤波器组的带宽序列
        # norm_factor = np.array(sigma_order1) / sigma_order1[-1]  # 正则化系数（各个滤波器组对应的带宽）
        # norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  # 正则化
        # # 非线性处理
        # Zxx = np.abs(norm_coefficients_order1)  # 取绝对值
        # quartiles = np.percentile(Zxx.flatten(), [25, 50, non_threshod])  # 四分位数 ideal95
        # # 超参数1的选择
        # origin_yvalue = 0.001
        # # 超参数2的选择
        # half_xvalue = quartiles[2] * 100.0
        # f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        # ZXX_non = Zxx * 100.0  # 标准化时频图,非线性函数的输入
        # norm_nonlinearprocess_coefficients_order1 = f(ZXX_non)  # 非线性处理
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        return coefficients_order1, t, freq_order1

    # 该方法返回正则化的一阶散射结果而不进行可视化
    def scattering_result_norm(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        time_duration = (T - 1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引

        # 使用重复采样方法将一阶散射的结果转换为可进行可视化的2d数组
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]  # 将不同子带内的小波系数扩展为相同的长度便于可视化
        coefficients_order1 = np.array(coefficients_order1)
        # # 对散射系数进行正则化处理
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]  # 一阶散射滤波器组的带宽序列
        norm_factor = np.log2(np.array(sigma_order1) / sigma_order1[-1] + 1)  # 正则化系数（各个滤波器组对应的带宽）
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  # 正则化
        # # 非线性处理
        # Zxx = np.abs(norm_coefficients_order1)  # 取绝对值
        # quartiles = np.percentile(Zxx.flatten(), [25, 50, non_threshod])  # 四分位数 ideal95
        # # 超参数1的选择
        # origin_yvalue = 0.001
        # # 超参数2的选择
        # half_xvalue = quartiles[2] * 100.0
        # f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        # ZXX_non = Zxx * 100.0  # 标准化时频图,非线性函数的输入
        # norm_nonlinearprocess_coefficients_order1 = f(ZXX_non)  # 非线性处理
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        return norm_coefficients_order1, t, freq_order1

    # 绘制经过非线性处理的SCWT视频图
    def scattering_result_nonlinear(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        time_duration = (T - 1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引

        # 使用重复采样方法将一阶散射的结果转换为可进行可视化的2d数组
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]  # 将不同子带内的小波系数扩展为相同的长度便于可视化
        coefficients_order1 = np.array(coefficients_order1)
        # # 对散射系数进行正则化处理
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]  # 一阶散射滤波器组的带宽序列
        norm_factor = np.log2(np.array(sigma_order1) / sigma_order1[-1] + 1)  # 正则化系数（各个滤波器组对应的带宽）
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  # 正则化
        # # 非线性处理
        Zxx = np.abs(norm_coefficients_order1)  # 取绝对值
        quartiles = np.percentile(Zxx.flatten(), [25, 50, 98])  # 四分位数 ideal95
        # 超参数1的选择
        origin_yvalue = 0.001
        # 超参数2的选择
        half_xvalue = quartiles[2] * 100.0
        f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        ZXX_non = Zxx * 100.0  # 标准化时频图,非线性函数的输入
        norm_nonlinearprocess_coefficients_order1 = f(ZXX_non)  # 非线性处理
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        return norm_nonlinearprocess_coefficients_order1, t, freq_order1

    # 绘制经过低通滤波的散射系数的时频图
    def scattering_result_visualisation(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        # T = 0  # 表示不进行低通滤波
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=True, out_type='array')  # 'list'\'array'两种输出格式
        # scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        order0 = np.where(meta['order'] == 0)  # 0阶散射系数（原始信号低通滤波）
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引
        order2 = np.where(meta['order'] == 2)
        coefficients_order1 = scattering_coefficients[order1]  # 获取一阶散射系数的数组（时频图）
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        lama1 = n_freq_order1[lama1]
        index_psi1_filter = np.where(meta['xi'][:, 0] == lama1)  # 获取要分析的一阶系数对应滤波器的索引
        coefficients_order2 = scattering_coefficients[index_psi1_filter][1:]  # 获取二阶散射系数的数组
        n_freq_order2 = meta['xi'][index_psi1_filter][1:, 1]   # 获取频率轴信息
        freq_order2 = n_freq_order2 * fs  # 获取二阶散射频率轴信息
        t = np.linspace(0, time_duration, scattering_coefficients.shape[-1])  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        # 绘图
        fig1 = visuallize_2d_array(coefficients_order1, X=t, Y=freq_order1, method='p', title='Title')  # 绘制一阶时频图
        fig2 = visuallize_2d_array(coefficients_order2, X=t, Y=freq_order2, method='p', title='Title')  # 绘制指定一阶尺度的二阶频率循环频率图

    # 绘制不经过低通滤波的散射系数的时频图
    def no_lowpass_scattering_result_visualisation(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        # order0 = np.where(meta['order'] == 0)  # 0阶散射系数（原始信号低通滤波）
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引
        # order2 = np.where(meta['order'] == 2)
        # 使用重复采样方法将一阶散射的结果转换为可进行可视化的2d数组
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]
        coefficients_order1 = np.array(coefficients_order1)
        # 对散射系数进行正则化处理
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]  # 一阶散射滤波器组的带宽序列
        norm_factor = np.array(sigma_order1)/sigma_order1[-1]  # 正则化系数（各个滤波器组对应的带宽）
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  # 正则化
        # 非线性处理
        Zxx = np.abs(norm_coefficients_order1)  # 取绝对值
        quartiles = np.percentile(Zxx.flatten(), [25, 50, 95])  # 四分位数 ideal95
        # 超参数1的选择
        origin_yvalue = 0.001
        # 超参数2的选择
        half_xvalue = quartiles[2] * 100.0
        f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        ZXX_non = Zxx * 100.0  # 标准化时频图,非线性函数的输入
        norm_nonlinearprocess_coefficients_order1 = f(ZXX_non)  # 非线性处理
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        # lama1 = n_freq_order1[lama1]
        # index_psi1_filter = np.where(meta['xi'][:, 0] == lama1)  # 获取要分析的一阶系数对应滤波器的索引
        # coefficients_order2 = scattering_coefficients[index_psi1_filter][1:]  # 获取二阶散射系数的数组
        # n_freq_order2 = meta['xi'][index_psi1_filter][1:, 1]   # 获取频率轴信息
        # freq_order2 = n_freq_order2 * fs  # 获取二阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        # 绘图
        fig1 = visuallize_2d_array(norm_coefficients_order1, X=t, Y=freq_order1, method='p', title='Title')  # 绘制一阶时频图
        # fig2 = visuallize_2d_array(norm_nonlinearprocess_coefficients_order1 , X=t, Y=freq_order1, method='p', title='Title')  # 绘制一阶时频图
        # fig2 = visuallize_2d_array(coefficients_order2, X=t, Y=freq_order2, method='p', title='Title')  # 绘制指定一阶尺度的二阶频率循环频率图
        # resize_acc_cwt_coefs = resize(norm_coefficients_order1, output_shape=(224, 224),
        #                               anti_aliasing=True)  # 尺寸校准# 数组插值实现统一的尺寸
        # acc_cwt_Fig_2d = visuallize_2d_array(resize_acc_cwt_coefs, X=t, Y=freq_order1, method='i', title='Title')  # 可视化
        print('1')

    def fig_save_no_lowpass_scattering_result_visualisation(self, signal, fs, lama1=None):
        '''
        该函数用来可视化小波散射的结果，包括可视化1阶散射的结果和指定分析频率下（一阶尺度，中心频率）的二阶散射的结果
        Parameters
        ----------
        signal ：待分析信号
        fs： 采样频率
        lama1 ：一阶散射小波滤波器的索引
        Returns
        -------
        绘制一阶2d时频图，指定分析频率的二阶时间、循环频率图
        '''
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1]  # 进行低通滤波时窗口的长度
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=True, out_type='list')  # 'list'\'array'两种输出格式
        scattering_coefficients = scattering(signal)  # 执行小波散射
        meta = scattering.meta()  # 获取每个节点的信息
        # order0 = np.where(meta['order'] == 0)  # 0阶散射系数（原始信号低通滤波）
        order1 = np.where(meta['order'] == 1)  # 1阶散射系数索引
        # order2 = np.where(meta['order'] == 2)
        # 使用重复采样方法将一阶散射的结果转换为可进行可视化的2d数组
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]
        coefficients_order1 = np.array(coefficients_order1)
        # 对散射系数进行正则化处理
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]]  # 一阶散射滤波器组的带宽序列
        norm_factor = np.array(sigma_order1)/sigma_order1[-1]  # 正则化系数（各个滤波器组对应的带宽）
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  # 正则化
        # 非线性处理
        Zxx = np.abs(norm_coefficients_order1)  # 取绝对值
        quartiles = np.percentile(Zxx.flatten(), [25, 50, 95])  # 四分位数 ideal95
        # 超参数1的选择
        origin_yvalue = 0.001
        # 超参数2的选择
        half_xvalue = quartiles[2] * 100.0
        f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
        ZXX_non = Zxx * 100.0  # 标准化时频图,非线性函数的输入
        norm_nonlinearprocess_coefficients_order1 = f(ZXX_non)  # 非线性处理
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        # lama1 = n_freq_order1[lama1]
        # index_psi1_filter = np.where(meta['xi'][:, 0] == lama1)  # 获取要分析的一阶系数对应滤波器的索引
        # coefficients_order2 = scattering_coefficients[index_psi1_filter][1:]  # 获取二阶散射系数的数组
        # n_freq_order2 = meta['xi'][index_psi1_filter][1:, 1]   # 获取频率轴信息
        # freq_order2 = n_freq_order2 * fs  # 获取二阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        # 绘图
        # fig1 = visuallize_2d_array(norm_coefficients_order1, X=t, Y=freq_order1, method='p', title='Title')  # 绘制一阶时频图
        # fig2 = visuallize_2d_array(norm_nonlinearprocess_coefficients_order1 , X=t, Y=freq_order1, method='p', title='Title')  # 绘制一阶时频图
        # 图像存储
        file_name = 'WHZZ_normal_wst' + '1' + '.png'
        file_path = r'F:\OneDrive - hnu.edu.cn\项目\学习记录文档\02瞬态冲击检测器改进（小波）2024.7\降噪瞬态冲击检测器\论文不同部分的插图\观点1'
        visual_2d_and_save(coefficients_order1, file_name, file_path, X=t, Y=freq_order1, fig_size=(3.2, 2.4), dpi=300, show=True, method='p')
        # fig2 = visuallize_2d_array(coefficients_order2, X=t, Y=freq_order2, method='p', title='Title')  # 绘制指定一阶尺度的二阶频率循环频率图
        # resize_acc_cwt_coefs = resize(norm_coefficients_order1, output_shape=(224, 224),
        #                               anti_aliasing=True)  # 尺寸校准# 数组插值实现统一的尺寸
        # acc_cwt_Fig_2d = visuallize_2d_array(resize_acc_cwt_coefs, X=t, Y=freq_order1, method='i', title='Title')  # 可视化
        print('1')
    # 绘制小波函数的实部和虚部,小波的函数的频域可视化
    def scattering_wavelet_visualisation(self, x):
        N = self._N_padded
        phi_f, psi1_f, psi2_f = scattering_filter_factory(N, self.J, self.Q, )
        # 一阶散射滤波器频域可视化
        plt.figure()
        plt.rcParams.update({"text.usetex": False})  # 不使用latex编码
        plt.plot(np.arange(T) / T, phi_f['levels'][0], 'r')  # 绘制低通滤波器的频域表示
        for psi_f in psi1_f:
            plt.plot(np.arange(T) / T, psi_f['levels'][0], 'b')  # 绘制一阶散射使用的小波函数
        plt.xlim(0, 0.5)
        plt.xlabel(r'$\omega$', fontsize=18)
        plt.ylabel(r'$\hat\psi_j(\omega)$', fontsize=18)
        plt.title('Frequency response of first-order filters (Q = {})'.format(Q),
                  fontsize=12)
        plt.show()
        # 二阶散射滤波器频域可视化
        plt.figure()
        plt.rcParams.update({"text.usetex": False})
        plt.plot(np.arange(T) / T, phi_f['levels'][0], 'r')
        for psi_f in psi2_f:
            plt.plot(np.arange(T) / T, psi_f['levels'][0], 'b')
        plt.xlim(0, 0.5)
        plt.ylim(0, 1.2)
        plt.xlabel(r'$\omega$', fontsize=18)
        plt.ylabel(r'$\hat\psi_j(\omega)$', fontsize=18)
        plt.title('Frequency response of second-order filters (Q = 1)', fontsize=12)
        # 滤波器时域可视化
        plt.figure()
        plt.rcParams.update({"text.usetex": False})
        psi_time = np.fft.ifft(psi1_f[-1]['levels'][0])
        psi_real = np.real(psi_time)
        psi_imag = np.imag(psi_time)
        plt.plot(np.concatenate((psi_real[-2 ** 8:], psi_real[:2 ** 8])), 'b')
        plt.plot(np.concatenate((psi_imag[-2 ** 8:], psi_imag[:2 ** 8])), 'r')
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$\psi(t)$', fontsize=18)
        plt.title('First-order filter - Time domain (Q = {})'.format(Q), fontsize=12)
        plt.legend(["$\psi$_real", "$\psi$_imag"])
        plt.show()



if __name__ == '__main__':
    file_path = (r'F:\OneDrive - hnu.edu.cn\项目\论文代码\小波散射网络及其改进和应用\小波散射网络代码\\'
                 r'wavelet_scattering_transform\data\free-spoken-digit-dataset\recordings\0_george_0.wav')  # zero发音音频文件
    fs, x = scipy.io.wavfile.read(file_path)  # 采样频率8K, 信号长度
    x = x / np.max(np.abs(x))  # 归一化
    x = torch.from_numpy(x).float()
    T = x.shape[-1]
    J = 6   # 最大尺度数
    Q = (16, 1)  # 每阶频程包含16个滤波器
    wavelet_scater = wavelet_scaterring_analysis(J, Q)  # 实例化小波散射对象
    wavelet_scater.scattering_result_visualisation(x, 8e3, lama1=2)
    phi_f, psi1_f, psi2_f = wavelet_scater.generate_filters(2**13)
    scattering_coefficients, meta = wavelet_scater.standard_scattering(x)
    wavelet_scater.scattering_wavelet_visualisation(x)
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)
    plt.figure()
    plt.rcParams.update({"text.usetex": False})
    plt.plot(np.arange(T) / T, phi_f['levels'][0], 'r')

    for psi_f in psi1_f:
        plt.plot(np.arange(T) / T, psi_f['levels'][0], 'b')

    # plt.xlim(0, 0.5)

    plt.xlabel(r'$\omega$', fontsize=18)
    plt.ylabel(r'$\hat\psi_j(\omega)$', fontsize=18)
    plt.title('Frequency response of first-order filters (Q = {})'.format(Q),
              fontsize=12)
    plt.show()
    print(1)