'''
该脚本是验证不同自适应滤波算法的主程序，该程序保留创建模拟信号和调用不同滤波方法的接口.
测试用例：
LMS:系统辨识（已知含噪信号与滤波后的理想信号，求解最优滤波器）
RLS:回声消除
todo: 自适应线性预测、自适应信道均衡。。。。
支持自适应滤波算法：
LMS:最小均方算法
NLMS:归一化最小均方算法
RLS:递归最小二乘算法
'''
import argparse   # 命令行解析器
from signal_process.adaptive_filter_signal_process import Adaptive_Filter
from utils.signal_generate import Signal_Generater
from utils.plot_utils import PlotAnalyzer
import matplotlib.pyplot as plt
import soundfile as sf

# 超参数设置
def parse_args():
    # 模拟信号生成参数
    parser = argparse.ArgumentParser(description='sample_analysis')  # 实例化参数解析器
    parser.add_argument('--sig_len', type=int, default=100, help='the length of signal in 10s')

    parser.add_argument('--random_seed', type=int, default=91, help='the random_seed')

    # LMS算法超参数调整
    parser.add_argument('--mu', type=int, default=0.1, help='the stride of iteration')
    parser.add_argument('--filter_order', type=int, default=8, help='the length of filter')



    # RLS算法超参数
    parser.add_argument('--lmbd', type=int, default=0.999, help='the weight factor')
    parser.add_argument('--delta', type=int, default=0.01, help='auto matrix')
    parser.add_argument('--N', type=int, default=256, help='the len fo filter')

    args = parser.parse_args()
    return args








if __name__ == '__main__':
    args = parse_args()  # 配置参数
    Signal_Ge = Signal_Generater(args) # 实例化模拟信号生成器
    Ada_Filter = Adaptive_Filter(args)  # 实例化自适应滤波器
    Ploter = PlotAnalyzer(args)  # 实例化绘图器

    # 模拟信号的获取
    # clean_signal, noisy_signal, t = Signal_Ge.ge_sine()  # 带噪声的正弦波
    x, d = Signal_Ge.ge_real()  # 实际语音信号(前者为噪声源(回声信号)、后者为麦克风接收到的含有噪声的信号)

    # 自适应滤波
    # filtered_signal, weights = Ada_Filter.lms_filter(noisy_signal, clean_signal)
    e = Ada_Filter.rls_filter(x, d)  # 返回不含有回声的信号

    sf.write('samples/rls.wav', e, int(8e3), subtype='PCM_16')
    # 滤波结果可视化
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.7)
    # plt.plot(t, filtered_signal, label='Filtered Signal (LMS)', color='red')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Noisy Signal and LMS Filtered Signal')
    # plt.show()

    print('finished')