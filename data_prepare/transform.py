import numpy as np
import torch
import cv2
import random
from scipy.signal import resample
from PIL import Image
import scipy
from skimage.transform import resize
from kymatio.torch import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from io import BytesIO
from PIL import Image
import math
from data_prepare.utils import *
import matplotlib.pyplot as plt
from scipy import signal
# from plot.plot_utils import visuallize_2d_array

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        seq = seq[np.newaxis, :]
        return seq

class Reshape_single_sample(object):
    def __call__(self, seq):
        seq = seq[np.newaxis, :]
        return seq[np.newaxis, :]

class Retype(object):
    def __call__(self, seq):
        seq = seq.astype(np.float32)
        seq = torch.from_numpy(seq)
        return seq

class ReSize(object):
    def __init__(self, size=1):
        self.size = size
    def __call__(self, seq):
        #seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)
        # seq = resize(seq, output_shape=(self.size, self.size))
        seq = resize(seq, output_shape=(self.size, self.size), anti_aliasing=True)
        seq = seq / 255
        return seq

class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        return seq*scale_factor


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):  # 随机返回0或者1
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
            return seq*scale_factor

class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_height = seq.shape[1] - self.crop_len
            max_length = seq.shape[2] - self.crop_len
            random_height = np.random.randint(max_height)
            random_length = np.random.randint(max_length)
            seq[random_length:random_length+self.crop_len, random_height:random_height+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type="0-1"):  # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std":
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


class wavelet_scaterring_analysis:
    '''
    该对象用来建立一个小波散射对象，可执行各种散射操作
    J表示最大尺度数
    Q表示单倍频程包含小波个数
    '''
    def __init__(self, J, Q, fs=None):
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

    def scattering_result_visualisation_for_CAM(self, signal, fs, non_threshod=95):
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
        quartiles = np.percentile(Zxx.flatten(), [25, 50, non_threshod])  # 四分位数 ideal95
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
        # 原始图像
        # fig1, ax1 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # pcm = ax1.pcolormesh(t, freq_order1, np.abs(norm_coefficients_order1), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # ax1.axis('off')  # 移除坐标轴
        # plt.tight_layout(pad=0)  # 调整图形以移除白边
        # buf_original = BytesIO()  # 将图形保存到内存中的BytesIO对象
        # plt.savefig(buf_original, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close(fig1)  # 关闭图形以释放内存
        # buf_original.seek(0)  # 将BytesIO对象转换为Image对象
        # original_img = Image.open(buf_original)
        # plt.imshow(original_img)
        # plt.axis('off')  # 去掉坐标轴
        # plt.show()

        # 非线性处理的图像
        fig2, ax2 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm1 = ax2.pcolormesh(t, freq_order1, np.abs(norm_nonlinearprocess_coefficients_order1), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ax2.axis('off')  # 移除坐标轴
        plt.tight_layout(pad=0)  # 调整图形以移除白边
        buf_non = BytesIO()  # 将图形保存到内存中的BytesIO对象
        plt.savefig(buf_non, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)  # 关闭图形以释放内存
        buf_non.seek(0)  # 将BytesIO对象转换为Image对象
        processed_img = Image.open(buf_non)
        # plt.imshow(processed_img)
        # plt.axis('off')  # 去掉坐标轴
        # plt.show()

        return 1, processed_img

    def cwt_result_visualisation_for_CAM(self, signal, fs, non_threshod=95):
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
        # 获取横轴和纵轴
        n_freq_order1 = meta['xi'][order1][:, 0]  # 获取一阶散射频率轴信息
        freq_order1 = n_freq_order1 * fs  # 获取一阶散射频率轴信息
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  # 获取时间轴信息,指定数组的开始时刻和结束时刻，以及数组的长度，生成时间轴
        # 原始图像
        fig1, ax1 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm = ax1.pcolormesh(t, freq_order1, np.abs(coefficients_order1), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ax1.axis('off')  # 移除坐标轴
        plt.tight_layout(pad=0)  # 调整图形以移除白边
        buf_original = BytesIO()  # 将图形保存到内存中的BytesIO对象
        plt.savefig(buf_original, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)  # 关闭图形以释放内存
        buf_original.seek(0)  # 将BytesIO对象转换为Image对象
        original_img = Image.open(buf_original)
        plt.imshow(original_img)
        plt.axis('off')  # 去掉坐标轴
        plt.show()

        # 2d数组可视化
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 2), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm = ax2.imshow(np.abs(coefficients_order1), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None,
                        vmax=None,
                        origin=None, extent=None)
        ax2.axis('off')  # 移除坐标轴
        plt.tight_layout(pad=0)  # 调整图形以移除白边
        buf_2darray = BytesIO()  # 将图形保存到内存中的BytesIO对象
        plt.savefig(buf_2darray, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)  # 关闭图形以释放内存
        buf_2darray.seek(0)  # 将BytesIO对象转换为Image对象
        array2d_img = Image.open(buf_2darray)
        plt.imshow(array2d_img)
        plt.axis('off')  # 去掉坐标轴
        plt.show()


        return original_img, array2d_img



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

        return f, t, amp

    # 获取stft+非线性处理的数组
    def stft_non_results_obtain(self, x):
        w_len = self.nperseg
        overlap = self.noverlap
        f, t, amp = signal.stft(x, self.fs, nperseg=w_len, noverlap=overlap)
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









