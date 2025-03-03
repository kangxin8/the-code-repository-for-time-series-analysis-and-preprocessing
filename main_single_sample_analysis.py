'''
该脚本旨在编写一个样本分析脚本，该脚本的特点是使用配置参数管理所有的参数，支持任何数据集的读取，分析和分析结果的保存。
使用流程：
1、在参数配置函数中，设置公共参数，主要是指定样本长度，当前分析的数据集,样本预处理方式
2、在预处理方法中设置要使用的信号处理方法的参数,主要是指定采样频率等信息
3、在相应的数据集上调整preprocess_params_dic
4、设置图像最终保存的文件夹
'''

import argparse
import os

import cv2
import torch
import numpy as np
from data_prepare.CWRU import CWRU
from data_prepare.MFPT import MFPT
from data_prepare.PU import PU
from data_prepare.PHM2024 import PHM2024
from data_prepare.PHM2009 import PHM2009
from data_prepare.XJTU import XJTU
from data_prepare.FRAHOF import FRAHOF
from data_prepare.UberU import UberU
from data_prepare.IMS import IMS
from data_prepare.JLXG import JLXG
from signal_process.signal_2d_transform_and_analysis import wavelet_scaterring_analysis, wavelet_scat_embeded_block, STFT_TF_analysis
from signal_process.signal_1d_transform_and_analysis import *
from utils.plot_utils import PlotAnalyzer
from skimage.transform import resize
import scipy.signal as sig

class Data_reader(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.sample_obj
        self.data_file_name = args.data_file_name
        self.sample_len = args.sample_len

        match self.dataset_name:
            case 'CWRU':
                self.dataset = CWRU(args)
            case 'MFPT':
                self.dataset = MFPT(args)
            case 'PU':
                self.dataset = PU(args)
            case 'PHM2024':
                self.dataset = PHM2024(args)
            case 'XJTU':
                self.dataset = XJTU(args)
            case 'FRAHOF':
                self.dataset = FRAHOF(args)
            case 'UberU':
                self.dataset = UberU(args)
            case 'PHM2009':
                self.dataset = PHM2009(args)

            case 'IMS':
                self.dataset = IMS(args)

            case 'JLXG':
                self.dataset = JLXG(args)



    def single_sample_read(self, sample_index=5):
        sample = self.dataset.single_file_sample_obtain(sample_index)
        sample_label = self.data_file_name
        return sample, sample_label

    # TODO: 样本列表的获取
    def sample_list_read(self, file_neme=None):
        sample_list, label_list = self.dataset.single_file_samplelist()
        return sample_list, label_list

    # TODO: dataloader的获取


class Data_analyzer(object):
    def __init__(self,args, sp_method=None, fs=None, fr=None, J=None, Q=None, nperseg=None, noverlap=None):
        self.fs = fs
        self.fr = fr
        self.args = args


    def signal_norm(self, seq, norm_method='0-1'):
        if norm_method == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif norm_method == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif norm_method == "mean-std":
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


    def scwt_signal_process(self, x):
        Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=self.args.scwt_J, Q=self.args.scwt_Q)  # 实例化小波散射对象
        ana_result, x, y = Wavelet_scatter_analysis.scattering_result(x, fs=self.args.fs)  # 获取小波散射时频图
        return ana_result, x, y


    def scwt_signal_process_non(self, x):
        Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=self.args.scwt_J, Q=self.args.scwt_Q)  # 实例化小波散射对象
        ana_result, x, y = Wavelet_scatter_analysis.scattering_result_nonlinear(x, fs=self.args.fs)  # 获取小波散射时频图
        return ana_result, x, y



    def scwt_signal_process_norm(self, x):
        Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=self.args.scwt_J, Q=self.args.scwt_Q)  # 实例化小波散射对象
        ana_result, x, y = Wavelet_scatter_analysis.scattering_result_norm(x, fs=self.args.fs)  # 获取小波散射时频图
        return ana_result, x, y



    def scwt_SE_attention_signal_process(self, x):

        x = torch.tensor(x)
        x = x.unsqueeze(0).unsqueeze(0)
        model = wavelet_scat_embeded_block(J=self.args.scwt_J, Q=self.args.scwt_Q)
        result = model(x)
        return result


    def stft_non_tf_signal_process(self, x):
        STFT_analysis = STFT_TF_analysis(fs=self.args.fs, nperseg=self.args.STFT_nperseg, noverlap=self.args.STFT_noverlap, non_threshod=self.args.non_threshod)  #
        f, t, stft_ZXX_non = STFT_analysis.stft_non_tf_results_obtain(x)
        return f, t, stft_ZXX_non


    def stft_signal_process(self, x):
        STFT_analysis = STFT_TF_analysis(fs=self.args.fs, nperseg=self.args.STFT_nperseg, noverlap=self.args.STFT_noverlap, non_threshod=self.args.non_threshod)  #
        f, t, amp = STFT_analysis.stft_results_obtain(x)
        return f, t, amp


    def plain_fft(self, x):
        Analysiser_1d = Signal_1d_Processer(self.args)
        fre, norm_result = Analysiser_1d.plain_fft_transform(x, sample_rate=self.args.fs)
        return fre, norm_result

    def fft_signal_process(self, x):
        Analysiser_1d = Signal_1d_Processer(self.args)
        fre, norm_result = Analysiser_1d.average_FFT_spec(x, samplingRate=self.args.fs)
        return fre, norm_result



    def acc_to_velsignal(self, data, sample_rate, filter_freq=10):
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
        sos = sig.butter(4, filter_freq, 'highpass', output='sos', fs=sample_rate)  # 高通滤波器
        data = sig.sosfilt(sos, data)  # 这里采用的是sos输出
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
        vel_res = sig.detrend(vel)  # 去除趋势项
        vel_res = vel_res * 9.8 * 1000  # 单位转换为mm
        return vel_res


def parse_args():
    parser = argparse.ArgumentParser(description='sample_analysis')  # 实例化参数解析器
    # ======分析对象公共参数========
    parser.add_argument('--sample_obj', type=str, default='CWRU', help='the dataset to which the current analysis sample belongs')
    # rightaxlebox/gearbox
    # N09_M07_F10_K001_1
    # N15_M07_F04_K001_1
    # N15_M01_F10_K001_1
    # N15_M07_F10_K001_1
    parser.add_argument('--data_file_name', type=str, default='136', help='the data file for analysis')
    parser.add_argument('--sample_len', type=int, default=4096, help='the len of single simple')  # 样本的长度
    parser.add_argument('--sample_preprocess', type=str, default='original_1d', help='the preprocess method for sample obtaining')
    parser.add_argument('--health_state', type=str, default='Lossness')

    # ========preprocess公共参数==========
    parser.add_argument('--fs', type=int, default=25e3, help='the fs for signal')
    # STFT超参数
    parser.add_argument('--STFT_nperseg', type=int, default=256, help='the window_size for STFT')
    parser.add_argument('--STFT_noverlap', type=int, default=255, help='the overlap for STFT')
    # SCWT超参数
    parser.add_argument('--scwt_J', type=int, default=10, help='the window_size for STFT')
    parser.add_argument('--scwt_Q', type=int, default=11, help='the overlap for STFT')
    # 非线性处理
    parser.add_argument('--non_threshod', type=int, default=95, help='the threshold for nonlinear_process')
    # 图像保存
    parser.add_argument('--save_dir', type=str,default=r'E:\datasets\凯斯西储大学数据\分析结果图像',help='the save dir for analysis')

    # ======不同数据集私有参数=========

    # CWRU
    parser.add_argument('--CWRU_fr', type=int, nargs='+', default=[1797, 1772, 1750, 1730], help='the rpm for 0,1,2,3hp condition')
    parser.add_argument('--CWRU_fs', nargs='+', default=[12e3, 48e3], help='the fs for CWRU')
    parser.add_argument('--CWRU_dic_index', type=int, default=3, help='the end for CWRU')

    # MFPT
    parser.add_argument('--MFPT_dic_index', type=int, default=3, help='the end for MFPT')

    # PU

    # PHM2024
    # 指定健康状态
    # 指定传感器名称
    parser.add_argument('--PHM2024_channel', type=str, default=['CH20'], help='the selected channel for location')

    # FRAHOF
    parser.add_argument('--FRAHOH_fs', type=int, default=4096, help='the fs for FRAHOH')
    parser.add_argument('--FRAHOH_dic_index', type=str, default=['Vibration_1'], help='the end for FRAHOH')

    # UberU
    parser.add_argument('--UberU_fs', type=int, default=25e3, help='the fs for UberU')
    parser.add_argument('--UberU_dic_index', type=int, default=2, help='the channel for UberU')




    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()  # 配置参数
    root_dir = r'D:\OneDrive - hnu.edu.cn\项目\学习记录文档\02瞬态冲击检测器改进（小波）2024.7\降噪瞬态冲击检测器\论文投稿后的返修\第一次返修绘图\统计直方图和非线性处理'
    save_dir = os.path.join(root_dir, args.data_file_name)  #
    # save_dir = os.path.join(root_dir, args.health_state)  #
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    # =======单个样本的获取========
    DATA_reader = Data_reader(args)
    # reault_data = DATA_reader.single_sample_read(sample_index=5)
    sample_list, label_list = DATA_reader.sample_list_read()

    # ====样本的变换和处理=======
    # signal_list = [sample_list[0], sample_list[10], sample_list[-1]]
    signal_list = [sample_list[1]]  # 10,4
    for signal in signal_list:
        # signal = sample_list[2000]
        Analyzer = Data_analyzer(args)
        # fft_plain, fft_amp_plain = Analyzer.plain_fft(signal)
        # vel_signal = Analyzer.acc_to_velsignal(data=signal, sample_rate=8e3)
        # fft_f, fft_amp = Analyzer.fft_signal_process(signal)
        # vel_fft_f, vel_fft_amp = Analyzer.fft_signal_process(vel_signal)
        stft_f, stft_t, stft_amp = Analyzer.stft_signal_process(signal)
        scwt_amp, scwt_x, scwt_y = Analyzer.scwt_signal_process(signal)
        # scwt_amp_norm, scwt_x_norm, scwt_y_norm = Analyzer.scwt_signal_process_norm(signal)
        scwt_amp_nonlinear, scwt_x_norm, scwt_y_norm = Analyzer.scwt_signal_process_non(signal)
        # resize_scwt_amp = resize(scwt_amp, (224, 224))
        # scwt_block_result = Analyzer.scwt_SE_attention_signal_process(signal)
        # stft_non_f, stft_non_t, stft_ZXX_non = Analyzer.stft_non_tf_signal_process(signal)
        # signal_norm = Analyzer.signal_norm(signal, norm_method='mean-std')
        # region散射分析


        # endregion

        # ======样本的可视化分析及保存=======
        Ploter = PlotAnalyzer(args)
        # Ploter.plot_1d_original_signal(signal, t=scwt_x, save=True, title=args.PHM2024_channel)
        # Ploter.plot_1d_signal_fft(amp=fft_amp, fre=fft_f, save=True, title=args.PHM2024_channel)
        # Ploter.plot_1d_signal_fft(amp=vel_fft_amp, fre=fft_f, save=True, title='vel')
        # Ploter.plot_1d_signal_fft(amp=fft_amp_plain, fre=fft_plain, save=True, title=args.CWRU_dic_index)
        Ploter.visualize_stft_tf(stft_amp, stft_t, stft_f, save=True, title='stft')
        Ploter.visualize_scwt_tf(scwt_amp, scwt_x, scwt_y, save=True, title='scwt')
        Ploter.visualize_scwt_tf(scwt_amp_nonlinear, scwt_x, scwt_y, save=True, title='scwt_nonlinear')
        # Ploter.visualize_scwt_tf(stft_non_f, stft_non_t, stft_ZXX_non, save=True, title='stft_nonlinear')
        # Ploter.visualize_resize_2darray(scwt_amp_norm, save=True, title='norm' + str(args.PHM2024_channel[0]))
        # Ploter.visualize_resize_2darray(scwt_amp, save=True, title=args.PHM2024_channel)
        # Ploter.visualize_tf(scwt_block_result, scwt_x, scwt_y, save=False)
        # Ploter.quick_visualize_2darray(scwt_block_result, save=False) 
        # Ploter.quick_visualize_2darray(scwt_block_result, save=False)
        # Ploter.plot_1d_original_signal(signal=signal_norm, save=True)
        print(1)




