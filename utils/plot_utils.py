import os

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from kymatio.torch import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator, MultipleLocator
from skimage.transform import resize
class PlotAnalyzer(object):
    def __init__(self, args, am=None, x=None, y=None):
        self.am = am
        self.x = x
        self.y = y
        self.args = args
    # 使用imshow或者matshow可视化二维数组
    def visualize_2darray(self, title=None, method='i'):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=100)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.imshow(np.abs(self.am), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None)
        # pcm = ax.matshow(np.abs(self.am), cfignum=None, cmap=None, norm=None, aspect=None)
        # 设置图形
        ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        ax.format_coord = format_coord
        plt.show()
        return fig

    # 使用pcolor可视化stft获取的时频图（使用插值）
    def visualize_stft_tf(self, amp, t, f, title=None, method='p', save=False):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.pcolormesh(t, f, np.abs(amp), cmap='viridis', norm=None, vmin=None, vmax=None,
                            shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # ax.axis('off')  # 取消刻度
        # 设置图形
        # ax.set_xlabel('Time [sec]')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_title(title)
        # # fig.colorbar(pcm, ax=ax)
        # plt.tight_layout()

        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'

        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_stft_tf-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

        return fig
    # 获取scwt获取的时频图
    def visualize_scwt_tf(self, amp, t, f, title=None, method='p', save=False):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.pcolormesh(t, f, np.abs(amp), cmap='viridis', norm=None, vmin=None, vmax=None,
                            shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # ax.axis('off')  # 取消刻度
        # 设置图形
        # ax.set_xlabel('Time [sec]')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_title(title)
        # # fig.colorbar(pcm, ax=ax)
        # plt.tight_layout()

        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'

        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_scwt_tf-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

        return fig

    # 绘制频谱
    def plot_1d_signal_fft(self,amp=None, fre=None, save=False, title=None):
        # 消除0频分量
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置,信号频谱和信号时域：(3.2,1.5),时频图（3.2，2.4）
        # self.am = self.am - np.mean(self.am)
        ax.plot(fre, amp, color='black', linewidth=1.25)
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.am), max(self.am) + 0.01)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        plt.show()
        if save:
            file_name = 'from-plot_1d_signal_fft-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    # 绘制原始振动信号
    def plot_1d_original_signal(self,signal=None, t=None, save=False, title=None):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置
        ax.plot(t, signal, color='black', linewidth=0.7)
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.y), max(self.y) + 0.0001)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        plt.show()
        if save:
            file_name = 'from-plot_1d_original_signal-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

    # 绘制不同通道的权重
    def plot_1d_attention(self, amp=None, fre=None, save=False, title=None):
        # 消除0频分量
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置,信号频谱和信号时域：(3.2,1.5),时频图（3.2，2.4）
        # self.am = self.am - np.mean(self.am)
        # ax.plot(fre, amp, color='black', linewidth=1.25)
        ax.bar(range(1, len(amp) + 1), amp,  color="skyblue", edgecolor="blue")
        ax.xticks(fontsize=5)
        ax.yticks(fontsize=5)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.am), max(self.am) + 0.01)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        plt.show()
        if save:
            file_name = 'from-plot_1d_signal_fft-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

    def img_save(self, img=None, file_path=None):

        pass
    def quick_visualize_2darray(self, array=None, title=None, method='i', save=False, figsize=None):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.imshow(np.abs(array), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None)
        # pcm = ax.matshow(np.abs(self.am), cfignum=None, cmap=None, norm=None, aspect=None)
        # 设置图形
        # ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-quick_visualize_2darray-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        return fig

    # 将原始矩形数组resize后可视化
    def visualize_resize_2darray(self, array=None, title=None, save=False, method='i', figsize=None):
        resize_array = resize(array, output_shape=(224, 224), anti_aliasing=True)  # 尺寸校准# 数组插值实现统一的尺寸
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        # pcm = ax.imshow(np.abs(resize_array), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None,
        #                 vmax=None, origin=None, extent=None)
        pcm = ax.matshow(np.abs(resize_array))
        # 设置图形
        # ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()

        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'

        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_resize_2darray-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        return fig




# 使用pcoloemesh可视化单个二维数组
# def visuallize_2d_array(C, X=None, Y=None, method='p', title='Title'):
#     '''
#     该函数用来实现指定方法的二维数组可视化
#     :param C: 要可视化的二维数组
#     :param X: 二维数组每个元素的横坐标
#     :param Y: 二维数组每个元素的纵坐标
#     :param method: 指定绘图函数imshow,pcolormesh,matshow.三者在是否插值、保持尺寸大小、横坐标标注方面有所区别
#     :param title: 图像的名称
#     :return:
#     返回绘制的图像
#     '''
#     fig, ax = plt.subplots(1, 1)
#     if method == 'p':
#         if X is None:
#             X = np.arange(C.shape[1] + 1)
#             Y = np.arange(C.shape[0] + 1)
#         fig, ax = plt.subplots(1,1)
#         pcm = ax.pcolormesh(X, Y, np.abs(C), cmap=None, norm=None, vmin=None, vmax=None, shading=None, alpha=None)
#         ax.set_xlabel('Time [sec]')
#         ax.set_ylabel('Frequency [Hz]')
#         ax.set_title(title)
#         fig.colorbar(pcm, ax=ax)
#         plt.tight_layout()
#         # plt.show()
#     elif method == 'i':
#         fig, ax = plt.subplots(1, 1)
#         pcm = ax.imshow(np.abs(C), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None,
#                    origin=None, extent=None)
#         ax.set_xlabel('Time [sec]')
#         ax.set_ylabel('Frequency [Hz]')
#         ax.set_title(title)
#         fig.colorbar(pcm, ax=ax)
#         plt.tight_layout()
#         # plt.show()
#     elif method == 'm':
#         fig, ax = plt.subplots(1, 1)
#         pcm = ax.matshow(np.abs(C), cfignum=None, cmap=None, norm=None, aspect=None)
#         ax.set_xlabel('Time [sec]')
#         ax.set_ylabel('Frequency [Hz]')
#         ax.set_title(title)
#         fig.colorbar(pcm, ax=ax)
#         plt.tight_layout()
#         # plt.show()
#     cursor = mplcursors.cursor(pcm, hover=True)
#     @cursor.connect("add")
#     def on_add(sel):
#         x, y = sel.target
#         z = sel.artist.get_array()[int(sel.target.index[0]), int(sel.target.index[1])]
#         sel.annotation.set_text(f'x: {x:.2f}s\nAmplitude: {y:.2f}\nIntensity: {z:.2f}')
#
#     plt.show()
#     return fig

def visual_1d_and_save(x_axes, y_axes, file_name, file_path, fig_size=(15, 5), dpi=300, show=True):
    '''
    该函数用来实现对1d频谱的绘制、存储和展示
    Parameters
    ----------
    x_axes
    y_axes
    file_name：文件名称
    file_path：文件存储的路径
    fig_size：绘制图像的大小和比例
    dpi：绘制的分辨率
    show：bool值，决定是否可视化展示
    Returns
    -------
    '''
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=300)  # 画布设置
    ax.plot(x_axes, y_axes, color='black', linewidth=1.5)
    # ax.axis('off')  # 取消刻度
    # 只显示纵轴的刻度和标签
    ax.yaxis.set_visible(True)
    ax.xaxis.set_visible(False)
    # 可选：设置纵轴刻度字体和大小
    # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
    plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
    ax.set_xlim(min(x_axes), max(x_axes))  # 设置显示的范围
    ax.set_ylim(min(y_axes), max(y_axes)+0.000001)
    # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
    # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
    # ax.spines['top'].set_color('none')  # 消除边框
    # ax.spines['right'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    if show is True:
        plt.show()
    save_path = os.path.join(file_path, file_name)
    fig.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)




def visual_2d_and_save(C, file_name, file_path, X=None, Y=None,  fig_size=(15, 5), dpi=300, show=True, method='p'):
    '''
    该函数用来实现对2d时频图的绘制、展示和存储
    Parameters
    ----------
    C
    file_name
    file_path
    X
    Y
    fig_size
    dpi
    show
    method

    Returns
    -------

    '''
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=300)   # 双栏（3.2， 2.4）dpi=300
    if method == 'p':
        if X is None:
            X = np.arange(C.shape[1] + 1)
            Y = np.arange(C.shape[0] + 1)
        pcm = ax.pcolormesh(X, Y, np.abs(C), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    elif method == 'i':
        pcm = ax.imshow(np.abs(C), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None,
                   origin=None, extent=None)
    elif method == 'm':
        pcm = ax.matshow(np.abs(C), cfignum=None, cmap=None, norm=None, aspect=None)
    # 图形格式设置
    formatter = ScalarFormatter(useMathText=True)  # 使用科学计数法
    formatter.set_powerlimits((0, 0))  # 强制以科学计数法显示
    # ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ax.tick_params(axis='both', which='major', labelsize=14)  # 设置图形刻度字体的大小
    plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
    plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)
    # ax.set_xlim(2, 8)  # 设置x轴的范围为2到8
    # ax.set_ylim(2, 8)  # 设置y轴的范围为2到8
    # 设置x轴和y轴的刻度数量
    # ax.xaxis.set_major_locator(MaxNLocator(5))  # x轴最多显示5个刻度
    # ax.yaxis.set_major_locator(MaxNLocator(3))  # y轴最多显示3个刻度
    # 或者使用 MultipleLocator 设置刻度间隔
    # ax.xaxis.set_major_locator(MultipleLocator())  # 每隔1个单位设置一个刻度
    # ax.yaxis.set_major_locator(MultipleLocator(2000))  # 每隔2个单位设置一个刻度
    # plt.tight_layout()
    if show is True:
        plt.show()
    # fig.colorbar(pcm, ax=ax)
    save_path = os.path.join(file_path, file_name)
    fig.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)




def visuallize_2d_array(C, X=None, Y=None, method='p', title='Title'):
    '''
    该函数用来实现指定方法的二维数组可视化
    :param C: 要可视化的二维数组
    :param X: 二维数组每个元素的横坐标
    :param Y: 二维数组每个元素的纵坐标
    :param method: 指定绘图函数imshow,pcolormesh,matshow.三者在是否插值、保持尺寸大小、横坐标标注方面有所区别
    :param title: 图像的名称
    :return:
    返回绘制的图像
    '''
    print(matplotlib.rcParams['figure.figsize'])  # 查看默认图形大小
    print(matplotlib.rcParams['figure.dpi'])  # 查看默认 DPI
    # 修改默认设置
    # plt.figure(figsize=(10, 6))  # 设置为 10 英寸 × 6 英寸
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=100)   # 双栏（3.2， 2.4）dpi=300
    if method == 'p':
        if X is None:
            X = np.arange(C.shape[1] + 1)
            Y = np.arange(C.shape[0] + 1)
        pcm = ax.pcolormesh(X, Y, np.abs(C), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    elif method == 'i':
        pcm = ax.imshow(np.abs(C), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None,
                   origin=None, extent=None)
    elif method == 'm':
        pcm = ax.matshow(np.abs(C), cfignum=None, cmap=None, norm=None, aspect=None)

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    plt.tight_layout()

    # cursor = mplcursors.cursor(pcm, hover=True)
    # @cursor.connect("add")
    # def on_add(sel):
    #     if method == 'p':
    #         x, y = sel.target
    #         row, col = int(sel.target.index[1]), int(sel.target.index[0])
    #     else:  # for 'i' and 'm'
    #         col, row = sel.target.index
    #         y, x = sel.target
    #     z = np.abs(C)[row, col]
    #     sel.annotation.set_text(f'Time: {x:.2f}s\nFrequency: {y:.2f}Hz\nIntensity: {z:.2f}')

    def format_coord(x, y):
        if method == 'p':
            col = int(x)
            row = int(y)
        else:  # for 'i' and 'm'
            col = int(x + 0.5)
            row = int(y + 0.5)

        if 0 <= row < C.shape[0] and 0 <= col < C.shape[1]:
            z = np.abs(C)[row, col]
            return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
        else:
            return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'

    ax.format_coord = format_coord
    plt.show()
    return fig

# 可视化单个1维数组
def visuallize_1d_array(t, signal, title='Title', Xlabel='Time (s)'):
    '''
    该函数用来实现对指定1d数组的可视化，并设置可视化游标悬停显示横纵坐标
    Parameters
    ----------
    t：横坐标刻度
    signal
    title ： 图形信息
    Xlabel：
    Returns
    -------
    '''
    fig, ax = plt.subplots(1, 1)
    line, = ax.plot(t, signal)
    # ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(Xlabel)
    ax.set_ylabel("Amplitude")
    # ax.set_xlim([0, t[-1]])
    # ax.set_ylim([0, signal.max()])
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    cursor = mplcursors.cursor(line, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f'x: {x:.2f}s\nAmplitude: {y:.2f}')
    plt.show()
    return fig

# 绘制双对数图和单对数图
def visuallize_1d_array1(t, signal, title='Title', Xlabel='Time (s)'):
    '''
    该函数用来实现对指定1d数组的可视化，并设置可视化游标悬停显示横纵坐标
    Parameters
    ----------
    t：横坐标刻度
    signal
    title ： 图形信息
    Xlabel：
    Returns
    -------
    '''
    fig, ax = plt.subplots(1, 1)
    line, = ax.loglog(t, signal)
    # ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(Xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([0, signal.max()])
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    cursor = mplcursors.cursor(line, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f'x: {x:.2f}s\nAmplitude: {y:.2f}')
    plt.show()
    return fig

# 绘制散点图
def plot_scatter(array, special_mark=None, x_mark_line=None, y_mark_line=None, title='Title'):
    fig, ax = plt.subplots(figsize=(10, 5))  # 返回图形对象和ax对象数组
    x = np.arange(len(array))  # 创建横坐标
    ax.scatter(x, array, label='decision')  # 绘制散点图
    ax.scatter(x[special_mark], array[special_mark], facecolors='none', edgecolors='r', label='Highlighted Points', s=100)  # 标记特殊散点
    ax.axhline(y=y_mark_line, color='r', linestyle='--', label='Reference Line')
    ax.axvline(x=x_mark_line, color='b', linestyle='--', linewidth=4,  label='Reference Line')
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('decision value')
    plt.title(title)
    plt.show()  # 显示图形
# 绘制多幅图像





