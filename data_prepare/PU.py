'''
该脚本用于以指定方式获取来自于PDBN的数据（单个数据样本、数据列表、dataloader）
关于PDBN数据的一些关键信息
1、采样频率为64K
2、转速及工况信息包含于文件名称
3、以mat文件存储，包含速度、温度、压力、振动等众多类型的信号
'''
import scipy.io as scio

from data_prepare.utils import *
from scipy.io import loadmat
import logging
from data_prepare.transform import wavelet_scaterring_analysis, STFT_TF_analysis
from itertools import chain
# Dataset:['CWRU', 'MFPT', 'IMS', 'PU']  #
# signal_process_method:['original_1d', 'STFT', 'SCWT']  # 支持的信号处理方法

# 该方法返回指定文件夹下所有文件形成的列表，列表包含四个元素，每个元素代表一种工况，包括20次重复采样获取的数据文件的绝对路径
def get_all_files_in_PUdirectory2(directory):
    """
    返回指定目录下所有文件的绝对路径，并形成一个列表
    Parameters:
    - directory: str, 要搜索的目录路径，eg:K005

    Returns:
    - file_paths: list, 包含所有文件绝对路径的列表
    """
    file_N_paths = []  # 存储正类样本
    file_F_paths = []  # 存储故障样本
    skip_files = ['K001.pdf', 'K002.pdf', 'K003.pdf', 'K004.pdf', 'K005.pdf', 'K006.pdf', 'KA01.pdf', 'KA03.pdf',
                  'KA04.pdf', 'KA05.pdf', 'KA06.pdf', 'KA07.pdf', 'KA08.pdf', 'KA09.pdf', 'KA15.pdf', 'KA16.pdf',
                  'KA22.pdf', 'KA30.pdf', 'KB23.pdf', 'KB24.pdf', 'KB27.pdf', 'KI01.pdf', 'KI03.pdf', 'KI04.pdf',
                  'KI05.pdf', 'KI07.pdf', 'KI08.pdf', 'KI14.pdf', 'KI16.pdf', 'KI17.pdf', 'KI18.pdf', 'KI21.pdf',
                  'measuring_log_K001.pdf', 'measuring_log_K002.pdf', 'measuring_log_K003.pdf', 'measuring_log_K004.pdf',
                  'measuring_log_K005.pdf', 'measuring_log_K006.pdf', 'measuring_log_KA01.pdf', 'measuring_log_KA03.pdf',
                  'measuring_log_KA04.pdf', 'measuring_log_KA05.pdf', 'measuring_log_KA06.pdf', 'measuring_log_KA07.pdf',
                  'measuring_log_KA08.pdf', 'measuring_log_KA09.pdf', 'measuring_log_KA15.pdf', 'measuring_log_KA16.pdf',
                  'measuring_log_KA22.pdf', 'measuring_log_KA30.pdf', 'measuring_log_KB23.pdf', 'measuring_log_KB24.pdf',
                  'measuring_log_KB27.pdf', 'measuring_log_KI01.pdf', 'measuring_log_KI03.pdf', 'measuring_log_KI04.pdf',
                  'measuring_log_KI05.pdf', 'measuring_log_KI07.pdf', 'measuring_log_KI08.pdf', 'measuring_log_KI14.pdf',
                  'measuring_log_KI16.pdf', 'measuring_log_KI17.pdf', 'measuring_log_KI18.pdf', 'measuring_log_KI21.pdf']
    pre_filename = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
    # 遍历目录及其子目录中的所有文件
    file_paths1 = []
    file_paths2 = []
    file_paths3 = []
    file_paths4 = []
    file_list = [file_paths1, file_paths2, file_paths3, file_paths4]
    for i in range(len(pre_filename)):
        for filename in os.listdir(directory):
            if filename.startswith(pre_filename[i]):
                filename = os.path.join(directory, filename)
                file_list[i].append(filename)

    return file_paths1, file_paths2, file_paths3, file_paths4


def wavelet_time_frequency(sample, fs=12e3, J=10, Q=(11, 1), non_threshod=95):
    '''
    该函数用来产生原始信号对应的时频图的两个PIL图形对象（原始时频图和非线性处理后的时频图）
    Parameters
    ----------
    sample：原始信号样本
    fs：采样频率
    J：最大尺度数（决定滤波器组的最低分析频率）
    Q：决定每倍频程的滤波器个数
    non_threshod：非线性的阈值
    Returns
    -------
    '''
    Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=J, Q=Q)  # 实例化小波散射对象
    original_img, processed_img = Wavelet_scatter_analysis.scattering_result_visualisation_for_CAM(sample, fs=fs, non_threshod=non_threshod)  # 获取小波散射时频图
    return original_img, processed_img

def CWT_time_frequency(sample, fs=12e3, J=10, Q=(11, 1), non_threshod=95):
    '''
    该函数用来产生原始信号对应的时频图的两个PIL图形对象（原始时频图和非线性处理后的时频图）
    Parameters
    ----------
    sample：原始信号样本
    fs：采样频率
    J：最大尺度数（决定滤波器组的最低分析频率）
    Q：决定每倍频程的滤波器个数
    non_threshod：非线性的阈值
    Returns
    -------
    '''
    Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=J, Q=Q)  # 实例化小波散射对象
    original_img, processed_img = Wavelet_scatter_analysis.cwt_result_visualisation_for_CAM(sample, fs=fs, non_threshod=non_threshod)  # 获取小波散射时频图
    return original_img, processed_img


def STFT_time_frequency(sample, fs=12e3, J=10, Q=(11, 1), non_threshod=95):
    '''
    该函数用来产生原始信号对应的时频图的两个PIL图形对象（原始时频图和非线性处理后的时频图）
    Parameters
    ----------
    sample：原始信号样本
    fs：采样频率
    J：最大尺度数（决定滤波器组的最低分析频率）
    Q：决定每倍频程的滤波器个数
    non_threshod：非线性的阈值
    Returns
    -------
    '''
    STFT_analysis = STFT_TF_analysis(fs=fs, nperseg=256, noverlap=250)  # 实例化小波散射对象
    processed_img = STFT_analysis.TFimage_obtain(sample)  # 获取小波散射时频图
    return processed_img

def STFT_result_array(sample, fs=12e3, J=10, Q=(11, 1),nperseg=None, noverlap=None, non_threshod=95):
    '''
    该函数用来产生原始信号对应的时频图的两个PIL图形对象（原始时频图和非线性处理后的时频图）
    Parameters
    ----------
    sample：原始信号样本
    fs：采样频率
    J：最大尺度数（决定滤波器组的最低分析频率）
    Q：决定每倍频程的滤波器个数
    non_threshod：非线性的阈值
    Returns
    -------
    '''
    STFT_analysis = STFT_TF_analysis(fs=fs, nperseg=nperseg, noverlap=noverlap, non_threshod=non_threshod)  #
    processed_array = STFT_analysis.stft_non_results_obtain(sample)  # 获取小波散射时频图
    return processed_array

# 返回指定文件名称的文件的绝对路径
def from_index_to_filepath(index='N09_M07_F10_K001_1', root_path=r'D:\德国帕德博恩轴承数据集\data'):
    '''
    该函数实现输入指定的数据文件名称，返回该文件的绝对路径
    Parameters
    ----------
    index ： 数据文件的名称
    root_path ： 整个数据集的根路径
    Returns
    -------
    指定索引的文件的绝对路径
    '''
    dir_list = read_obsolute_path_dir(root_path)  # 根目录下所有文件夹的绝对路径
    file_path_list = []
    for dir in dir_list:
        file_list = read_obsolete_path_file(dir)  # 获取文件夹下所有文件的绝对路径
        file_path_list.extend(file_list)
    result_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in file_path_list}  # 使用字典推导式创建所需的字典
    file_path = result_dict[index]
    return file_path
# 返回指定文件夹名称的文件夹的绝对路径
def from_dir_index_to_dirpath(index='K005', root_path=r'D:\德国帕德博恩轴承数据集\data'):
    '''
    该函数实现输入指定的数据文件名称，返回该文件的绝对路径
    Parameters
    ----------
    index ： 数据文件的名称
    root_path ： 整个数据集的根路径
    Returns
    -------
    指定索引的文件的绝对路径
    '''
    dir_list = read_obsolute_path_dir(root_path)  # 根目录下所有文件夹的绝对路径
    dir_path_dic = {os.path.splitext(os.path.basename(dir))[0]: dir for dir in dir_list}  # 使用推导式构建字典
    dir_path = dir_path_dic[index]
    # file_path_list = []
    # for dir in dir_list:
    #     file_list = read_obsolete_path_file(dir)  # 获取文件夹下所有文件的绝对路径
    #     file_path_list.extend(file_list)
    # result_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in file_path_list}  # 使用字典推导式创建所需的字典
    # file_path = result_dict[index]
    return dir_path
def filepath_to_samplelist(args, filepath, sample_len=2048, index=3, label=None, signal_process_method=None, params=None):
    '''
    该函数用来将PU数据集中的一个.mat文件的原始振动信号划分为样本，并返回一个由样本组成的列表
    filepath:文件的路径
    sample_len: 单个样本的长度
    label:类别标签
    返回：以列表形式返回
    '''
    data = scio.loadmat(filepath)  # 读取文件内容
    data_key_list = list(data.keys())  # 读取出的内容以字典形式存储,获取键的列表
    desired_key = data_key_list[3]  # 指定键的名称
    desired_data = data[desired_key][0, 0]['Y']
    fl = desired_data[0, 6]['Data'].flatten()

    # data = loadmat(filepath)  # 读取文件内容
    # data_key_list = list(data.keys())  # 读取出的内容以字典形式存储,获取键的列表
    # desired_key = data_key_list[index]  # 指定键的名称
    # fl = data[desired_key].flatten()
    fl = fl.reshape(-1,)
    data = []
    lab = []
    start, end = 0, args.sample_len
    while end <= fl.shape[0]:
        x = fl[start:end]
        match args.sample_preprocess:
            case 'original_1d':
                data.append(x)
                lab.append(label)

            case 'CWT':
                processed_tf_img, _ = CWT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                             non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)
            case 'SCWT':
                _, processed_tf_img = wavelet_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                         non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)
            case 'STFT':
                processed_tf_img = STFT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                         non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)

            case 'STFT_non':
                processed_result = STFT_result_array(x, fs=params.fs, J=params.J, Q=params.Q,
                                                     nperseg=params.nperseg, noverlap=params.noverlap,
                                                     non_threshod=params.non_threshod)
                # 非线性处理
                data.append(processed_result)
                lab.append(label)
            case 'fft':
                sub_data = np.fft.fft(x)
                sub_data = np.abs(sub_data) / len(sub_data)
                sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1, )
                sub_data = (sub_data - sub_data.min()) / (sub_data.max() - sub_data.min())  # 正则化方法
                data.append(sub_data)
                lab.append(label)
        start += args.sample_len
        end += args.sample_len
    return data, lab

def dataset_construct(args, files, sample_len=2048, index=3, label=1, signal_process_method='original_1d', SCWT_params=None):
    # 将文件名称列表转换为路径列表
    file_path_list = files  # 将文件名称列表转换为路径列表
    # 对每个列表分别处理
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist(args, file_path, sample_len=args.sam_len, index=args.dic_index,
                                           label=label, signal_process_method=signal_process_method, params=SCWT_params)
        samples_list.extend(data)
        labels_list.extend(lab)

    return samples_list, labels_list

class CWT_Params:
    def __init__(self, fs=12e3, J=10, Q=(11, 1), nperseg=256, noverlap=250, non_threshod=95):
        self.fs = fs
        self.J = int(J)
        self.Q = Q
        self.non_threshod = non_threshod
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __str__(self):
        return f'CWT_Params(fs={self.fs}, J={self.J}, Q={self.Q}, non_threshod={self.non_threshod})'
class DictObj:
    '''
    This class can convert a dict into a python class.
    Then we can access the attributes via ".keyname" instead of "[keyname]"
    '''
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])  # 学习这里递归的调用方法
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
class PU(object):
    def __init__(self,args, domain_name=None, sample_len=None, preprocess_name=None, file_name='97', single_domain=True, sub_dataset_name='a01', preprocess='STFT_non'):
        self.sample_len = args.sample_len

        preprocess_params_dic = {'dic_index': 3, 'label': 0,  # 样本参数
                                 'sam_len': self.sample_len, 'fs': 48e3, 'fr': 1797,
                                 'J': 10, 'Q': (11, 1),  # 小波变换参数
                                 'nperseg': 256, 'noverlap': 250,  # STFT参数
                                 'non_threshod': 95}
        self.preprocess_params = DictObj(preprocess_params_dic)
        self.domain_name = domain_name
        self.sample_len = sample_len
        self.file_name = file_name
        self.preprocess_name = preprocess_name
        self.single_domain = single_domain
        self.sub_dataset_name = sub_dataset_name
        self.preprocess = preprocess
        self.args = args
    # 获取指定文件名称的文件形成的样本列表
    def single_file_samplelist(self, label=1, file_info=None, channel=None):
        # 将文件名称列表转换为路径列表
        if file_info is None:
            file_list = [self.args.data_file_name]
        else:
            file_list = file_info
        file_path_list = [from_index_to_filepath(file) for file in file_list]  # 将文件名称列表转换为路径列表
        # 对每个列表分别处理
        samples_list = []  # 样本列表
        labels_list = []  # 标签列表
        for file_path in file_path_list:
            data, lab = filepath_to_samplelist(self.args, file_path, sample_len=self.args.sample_len)
            samples_list.extend(data)
            labels_list.extend(lab)

        return samples_list, labels_list


    # 获取由指定文件夹产生的样本列表
    def single_dir_samplelist(self, file_name):
        dir_list1 = [file_name]  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,64K,注意与其它数据集不同，PU数据集每种状态都重复测试多次
        dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in dir_list1]  # 获取文件夹的绝对路径
        file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in dir_list_path1]  # 获取文件夹下所有文件形成的列表
        Normal_sample_list = []
        Normal_label_list = []
        for i in range(len(file_list1)):
            for j in range(len(file_list1[i])):
                Sample_list, Label_list = dataset_construct(self.preprocess_params,files=file_list1[i][j], sample_len=self.sample_len, index=3,
                                                            label=0,
                                                            signal_process_method='original_1d')  # 将列表中的文件转换为样本
                Normal_sample_list.append(Sample_list)
                Normal_label_list.append(Label_list)
        sample_list = list(chain.from_iterable(Normal_sample_list))  # 拼接为一个完整的健康样本列表
        label_list = list(chain.from_iterable(Normal_label_list))

        return sample_list, label_list
    # 获取指定文件的单个样本
    def single_file_sample_obtain(self, sample_index=5):
        file_list = [self.file_name]
        sample_list, label_list = dataset_construct(files=file_list, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        sample = sample_list[sample_index]
        return sample
    # 获取指定文件夹产生的多个样本列表
    def single_task_sample_obtain(self, dir='K005', index=0):
        # 健康样本,标签：0
        dir_list1 = ['K005']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,64K,注意与其它数据集不同，PU数据集每种状态都重复测试多次
        dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in dir_list1]  # 获取文件夹的绝对路径
        file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in dir_list_path1]  # 获取文件夹下所有文件形成的列表
        Normal_sample_list = []
        Normal_label_list = []
        for i in range(len(file_list1)):
            for j in range(len(file_list1[i])):
                Sample_list, Label_list = dataset_construct(files=file_list1[i][j], sample_len=self.sample_len, index=3,
                                                            label=0, signal_process_method='original_1d')  # 将列表中的文件转换为样本
                Normal_sample_list.append(Sample_list)
                Normal_label_list.append(Label_list)
        return Normal_sample_list, Normal_label_list
    # 获取多个文件共同形成的样本列表
    def original_1d(self):

        # 健康样本,标签：0
        dir_list1 = ['K005']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,64K,注意与其它数据集不同，PU数据集每种状态都重复测试多次
        dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in dir_list1]  # 获取文件夹的绝对路径
        file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in dir_list_path1]  # 获取文件夹下所有文件形成的列表
        Normal_sample_list = []
        Normal_label_list = []
        for i in range(len(file_list1)):
            for j in range(len(file_list1[i])):
                Sample_list, Label_list = dataset_construct(files=file_list1[i][j], sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
                Normal_sample_list.append(Sample_list)
                Normal_label_list.append(Label_list)
        sample_list1 = list(chain.from_iterable(Normal_sample_list))  # 拼接为一个完整的健康样本列表
        label_list1 = list(chain.from_iterable(Normal_label_list))
        # sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=4096, index=3, label=0,
        #                                               signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=4096, index=6, label=0,
        #                                               signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))  # 记录日志
        # # 内圈故障 1
        # file_list3 = ['105', '106', '169', '170', '209', '210']  # 12K驱动端轴承内圈故障
        # sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=4096, index=3, label=1,
        #                                               signal_process_method='original_1d')
        # file_list4 = ['109', '110', '174', '175', '213', '214']  # 48K驱动端轴承内圈故障
        # sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=4096, index=3, label=1,
        #                                               signal_process_method='original_1d')
        # file_list5 = ['278', '279', '274', '275', '270', '271']  # 12K风扇端轴承内圈故障
        # sample_list5, label_list5 = dataset_construct(files=file_list5, sample_len=4096, index=4, label=1,
        #                                               signal_process_method='original_1d')
        # logging.info('number_of_IF_class:{}'.format(len(sample_list3 + sample_list4 + sample_list5)))  # 记录日志
        #
        # # 外圈故障 2
        # file_list6 = ['130', '131', '234', '235', '144', '145']  # 12K驱动端轴承外圈故障
        # sample_list6, label_list6 = dataset_construct(files=file_list6, sample_len=4096, index=3, label=2,
        #                                               signal_process_method='original_1d')
        # file_list7 = ['135', '136', '201', '202', '238', '239', '148', '149']  # 48K驱动端轴承外圈故障
        # sample_list7, label_list7 = dataset_construct(files=file_list7, sample_len=4096, index=3, label=2,
        #                                               signal_process_method='original_1d')
        # file_list8 = ['294', '295', '313', '315', '309', '310']  # 12K风扇端轴承外圈故障
        # sample_list8, label_list8 = dataset_construct(files=file_list8, sample_len=4096, index=4, label=2,
        #                                               signal_process_method='original_1d')
        # logging.info('number_of_OF_class:{}'.format(len(sample_list6 + sample_list7 + sample_list8)))  # 记录日志
        #
        # # 将所有的样本组合起来
        # sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4 + sample_list5 + sample_list6 + sample_list7 + sample_list8
        # label_list = label_list1 + label_list2 + label_list3 + label_list4 + label_list5 + label_list6 + label_list7 + label_list8
        # logging.info('number_of_total_samples:{}'.format(len(sample_list)))
        # endregion
        return sample_list1, label_list1
    def STFT(self):
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['97', '98', '100']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=95)
        # logging.info('SCWT_params:{}'.format(params))
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=4, label=1,
                                                      signal_process_method='STFT', SCWT_params=params)  # 将列表中的文件转换为样本
        file_list2 = ['99']
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1,
                                                      signal_process_method='STFT', SCWT_params=params)  # 将列表中的文件转换为样本
        # # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))
        # class2 故障类
        params3 = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=92)
        file_list3 = ['109', '175', '135']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='STFT', SCWT_params=params3)
        params4 = CWT_Params(fs=12e3, J=10, Q=(11, 1), non_threshod=92)
        file_list4 = ['105', '169', '209', '130', '197', '234', '223']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='STFT', SCWT_params=params4)
        logging.info('number_of_fault_class:{}'.format(len(sample_list3 + sample_list4)))
        # logging.info('测试数据文件:{}/{}/{}/{}'.format(file_list1, file_list2, file_list3, file_list4))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4
        # sample_list = sample_list3
        # label_list = label_list3
        label_list = label_list1 + label_list2 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list

    def CWT(self):
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['97', '98', '100']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=95)
        # logging.info('SCWT_params:{}'.format(params))
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=4, label=1,
                                                      signal_process_method='CWT', SCWT_params=params)  # 将列表中的文件转换为样本
        file_list2 = ['99']
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1,
                                                      signal_process_method='CWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))
        # class2 故障类
        params3 = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=92)
        file_list3 = ['109', '175', '135']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='CWT', SCWT_params=params3)
        params4 = CWT_Params(fs=12e3, J=10, Q=(11, 1), non_threshod=92)
        file_list4 = ['105', '169', '209', '130', '197', '234', '223']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='CWT', SCWT_params=params4)
        logging.info('number_of_fault_class:{}'.format(len(sample_list3 + sample_list4)))
        # logging.info('测试数据文件:{}/{}/{}/{}'.format(file_list1, file_list2, file_list3, file_list4))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4
        # sample_list = sample_list3
        # label_list = label_list3
        label_list = label_list1 + label_list2 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list
    def first_revised_stft_non(self):
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), nperseg=256, noverlap=250, non_threshod=95)

        a01 = ['K004']
        a02 = ['K004']
        a03 = ['K004']
        a04 = ['K004']
        a05 = ['KA05']
        a06 = ['KA05']
        a07 = ['KA05']
        a08 = ['KA05']
        a09 = ['KI07']
        a10 = ['KI07']
        a11 = ['KI07']
        a12 = ['KI07']
        if self.single_domain:
            # region 单一子数据集获取
            match self.sub_dataset_name:
                case 'a01':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a01]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][0],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=0, SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01,params.nperseg,params.noverlap,params.non_threshod )
                    )
                    return sample_list, label_list


                case 'a02':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a02]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][1],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=0,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a03':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a03]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][2],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=0,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a04':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a04]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][3],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=0,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a05':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a05]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][0],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a06':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a06]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][1],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a07':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a07]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][2],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a08':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a09]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][3],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a09':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a09]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][0],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a10':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a10]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][1],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a11':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a11]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][2],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a12':
                    dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in a12]  # 获取文件夹的绝对路径
                    file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in
                                  dir_list_path1]  # 获取文件夹下所有文件形成的列表
                    sample_list = []
                    label_list = []
                    for i in range(len(file_list1)):
                        Sample_list, Label_list = dataset_construct(files=file_list1[i][3],
                                                                    sample_len=self.sample_len, index=3,
                                                                    label=1,SCWT_params=params,
                                                                    signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                        sample_list.append(Sample_list)
                        label_list.append(Label_list)
                    sample_list = list(chain.from_iterable(sample_list))  # 拼接为一个完整的健康样本列表
                    label_list = list(chain.from_iterable(label_list))
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
            # endregion

    # 该方法用来获取PU数据集的总的测试集
    def first_revised_total(self):
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), nperseg=256, noverlap=250, non_threshod=95)
        # 健康样本,标签：0
        dir_list1 = ['K004', 'KA05', 'KI07']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,64K,注意与其它数据集不同，PU数据集每种状态都重复测试多次
        dir_list_path1 = [from_dir_index_to_dirpath(dir) for dir in dir_list1]  # 获取文件夹的绝对路径
        file_list1 = [get_all_files_in_PUdirectory2(dir_path) for dir_path in dir_list_path1]  # 获取文件夹下所有文件形成的列表
        Normal_sample_list = []
        Normal_label_list = []
        for i in range(len(file_list1)):
            for j in range(len(file_list1[i])):
                if i==0:
                    label = 0
                else:
                    label = 1
                Sample_list, Label_list = dataset_construct(files=file_list1[i][j], sample_len=self.sample_len, index=3, label=label,SCWT_params=params,
                                                      signal_process_method=self.preprocess)  # 将列表中的文件转换为样本
                Normal_sample_list.append(Sample_list)
                Normal_label_list.append(Label_list)
        sample_list1 = list(chain.from_iterable(Normal_sample_list))  # 拼接为一个完整的健康样本列表
        label_list1 = list(chain.from_iterable(Normal_label_list))
        return sample_list1, label_list1


    def SCWT(self):
        # class1 健康样本标签为1，故障样本为0
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['97', '98', '100']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=95)
        logging.info('SCWT_params:{}'.format(params))
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=4, label=1,
                                                      signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
        file_list2 = ['99']
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1,
                                                      signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))
        params3 = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=92)
        # class2 故障类
        file_list3 = ['109', '175', '135']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='SCWT', SCWT_params=params3)
        params4 = CWT_Params(fs=12e3, J=10, Q=(11, 1), non_threshod=92)
        file_list4 = ['105', '169', '209', '130', '197', '234', '223']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='SCWT', SCWT_params=params4)
        logging.info('number_of_fault_class:{}'.format(len(sample_list3+sample_list4)))
        logging.info('测试数据文件:{}/{}/{}/{}'.format(file_list1, file_list2, file_list3, file_list4))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4
        # sample_list = sample_list3
        # label_list = label_list3
        label_list = label_list1 + label_list2 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list


if __name__ == '__main__':
    Dataset = PU(sample_len=4096, sub_dataset_name='a01')  # 实例化数据集构造对象
    sam, lab = Dataset.first_revised_stft_non()
