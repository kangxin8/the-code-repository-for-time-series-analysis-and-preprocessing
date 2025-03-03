'''
该脚本编写一个PHM2024数据集数据读取对象，用于训练集、测试集的构建以及样本的分析

数据集基本信息
dir:E:\datasets\PHM\PHM2024\Preliminary_Stage\Datasets\Training\训练集
数据集说明文件：file:///E:/datasets/PHM/PHM2024/
fs:64K
Type:表示一种健康状态文件夹，该文件夹下包含若干样本，每个样本文件夹下包含4个excel文件，分别表示在牵引电机、减速齿轮箱和两个轴箱位置上安装的传感器采集的信号
Type0:健康状态，Type1：短路，Type2：电机转子条断裂，Type3：轴承故障，Type4：轴弯曲，Type5：齿根裂纹，Type6：齿面磨损，Type7：缺齿，Type8：断齿，
Type9：轴承内圈，Type10：轴承外圈，Type11：滚动体，Type12：保持架，Type13：轴承内圈，Type14：轴承外圈，Type15：滚动体，Type16：保持架

主要功能：
1、获取指定文件（传感器位置），指定通道（传感器方向）形成的样本列表
使用流程：
1、
'''
from data_prepare.utils import *
import pandas as pd
import argparse   # 命令行解析器


# 该方法用来将指定的文件名称列表转换为文件路径列表
def from_index_to_filepath(file_info, root_path=r'E:\datasets\PHM\PHM2024\Preliminary_Stage\Datasets\Training\训练集'):
    '''
    该函数实现输入指定的数据文件名称，返回该文件的绝对路径.PHM2024
    Parameters
    ----------
    index ： 数据文件的名称
    root_path ： 整个数据集的根路径
    Returns
    -------
    指定索引的文件的绝对路径
    '''
    dir_list = read_obsolute_path_dir(root_path)  # 根目录下所有文件夹(故障类型)的绝对路径
    file_path_list = []
    for dir in dir_list:
        sample_dir_list = read_obsolute_path_dir(dir)
        for sample_dir in sample_dir_list:
            file_list = read_obsolete_path_file(sample_dir)  # 获取文件夹下所有文件的绝对路径
            file_path_list.extend(file_list)
    result_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in file_path_list}  # 使用字典推导式创建所需的字典
    file_path = result_dict[file_info]
    return file_path

# 该函数用于对指定文件的数据以特定的形式形成列表
def filepath_to_samplelist(args, file=None, channel=None, label=None):
    '''
    该函数用来将CWRU数据集中的一个.mat文件的原始振动信号划分为样本，并返回一个由样本组成的列表
    filepath:文件的路径
    sample_len: 单个样本的长度
    label:类别标签
    返回：以列表形式返回
    '''
    data = pd.read_csv(file, usecols=channel)   # 读取数据
    fl = data.to_numpy().flatten()
    data = []
    lab = []
    start, end = 0, args.sample_len
    while end <= fl.shape[0]:
        x = fl[start:end]
        match args.sample_preprocess:
            case 'original_1d':
                data.append(x)
                lab.append(label)

            # case 'CWT':
            #     processed_tf_img, _ = CWT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
            #                                                  non_threshod=params.non_threshod)
            #     data.append(processed_tf_img)
            #     lab.append(label)
            # case 'SCWT':
            #     _, processed_tf_img = wavelet_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
            #                                              non_threshod=params.non_threshod)
            #     data.append(processed_tf_img)
            #     lab.append(label)
            # case 'STFT':
            #     processed_tf_img = STFT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
            #                                              non_threshod=params.non_threshod)
            #     data.append(processed_tf_img)
            #     lab.append(label)
                # f, t, imgs = stft_window(x, fs=fs, nperseg=nperseg, noverlap=noverlop)  # 按照指定的超参数做STFT变换，返回绝对值
                # quartiles = np.percentile(imgs.flatten(), [25, 50, threshhold])  # 四分位数
                # origin_yvalue = 0.001  # 超参数1的选择
                # half_xvalue = quartiles[2] * 100.0  # 超参数2的选择
                # f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
                # ZXX_non = imgs * 100.0  # 标准化时频图,非线性函数的输入
                # stft_ZXX_non = f(ZXX_non)  # 非线性处理
                # stft_ZXX_non_resize = resize(imgs, output_shape=(224, 224), anti_aliasing=True)  # 尺寸校准
        start += args.sample_len
        end += args.sample_len
    return data, lab




# 该函数完成对指定数据内容的读取，处理，形成列表
def dataset_construct(args, label=None, file_info=None, channel=None):
    '''

    Parameters
    ----------
    args
    label：每次只接受同类别的样本或者数据集的读取
    file_info:供自定义选择文件构造数据集的接口，是字典形式

    Returns
    -------

    '''
    # =======将待读取的文件名称列表转换为路径列表======
    if file_info is None:
        file_list = [args.data_file_name]
        Channel = args.PHM2024_channel
    else:
        file_list = file_info
        Channel = channel
    file_path_list = [from_index_to_filepath(file) for file in file_list]  # 将文件名称列表转换为路径列表
    # 读取文件形成列表
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist(args, file=file_path, channel=Channel, label=label)
        samples_list.extend(data)
        labels_list.extend(lab)

    return samples_list, labels_list






class PHM2024(object):
    def __init__(self, args):
        self.args = args

    #  返回指定文件，指定通道数据形成的列表
    def single_file_samplelist(self):

        sample_list, label_list = dataset_construct(self.args, label=1)  # 将列表中的文件转换为样本

        return sample_list, label_list



# 包含所有数据集和预处理方法的参数
def parse_args():
    parser = argparse.ArgumentParser(description='sample_analysis')  # 实例化参数解析器
    # ======分析对象公共参数========
    parser.add_argument('--sample_obj', type=str, default='PU', help='the dataset to which the current analysis sample belongs')  # 分析的数据集名称
    parser.add_argument('--data_file_name', type=str, default='TYPE0_Sample1_data_motor', help='the data file for analysis')  # KA16 KA22 KI16 KI17 分析的文件名称
    parser.add_argument('--sample_len', type=int, default=8192, help='the len of single simple')  # 样本的长度
    parser.add_argument('--sample_preprocess', type=str, default='original_1d', help='the preprocess method for sample obtaining')  # 样本的处理方式
    parser.add_argument('--analysis', type=bool, default=True, help='decision for analysis or dataset construction')  # 数据读取的用图，False,表示用于构造数据集而非分析单个样本

    # CWRU
    parser.add_argument('--CWRU_fr', type=int, nargs='+', default=[1797, 1772, 1750, 1730], help='the rpm for 0,1,2,3hp condition')
    parser.add_argument('--CWRU_fs', nargs='+', default=[12e3, 48e3], help='the fs for CWRU')

    # MFPT

    # PU

    # PHM2024
    # 指定健康状态
    # 指定传感器名称
    parser.add_argument('--PHM2024_channel', type=str, default=['CH1'], help='the selected channel for location')  # 指定通道名称(代表方向)

    # 元组形式的故障类型与传感器位置，通道之间的匹配，用于构造训练集或者测试集
    # .....

    # ========preprocess==========
    parser.add_argument('--fs', type=int,  default=64e3, help='the fs for signal')
    # STFT超参数
    parser.add_argument('--STFT_nperseg', type=int,  default=256, help='the window_size for STFT')
    parser.add_argument('--STFT_noverlap', type=int,  default=255, help='the overlap for STFT')
    # 非线性处理
    parser.add_argument('--non_threshod', type=int,  default=95, help='the threshold for nonlinear_process')
    # 图像保存
    parser.add_argument('--save_dir', type=str,
                        default=r'F:\OneDrive - hnu.edu.cn\项目\学习记录文档\01瞬态冲击检测器英文2024.6\可解释瞬态冲击现象检测器\论文投稿后返修\返修绘图\样本在不同域的直观展示\新增数据集',
                        help='the save dir for analysis')



    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()  # 配置参数
    Dataset = PHM2024(args)
    sample_list, label_list = Dataset.single_file_samplelist()
