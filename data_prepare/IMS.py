'''
该脚本用于将基于IMS构建的数据集以样本和标签的列表形式返回
关于IMS数据的一些关键信息
1、采样频率统一为20K,为全寿命周期数据，单个数据文件采样1s，样本的长度为20480，一个文件对应5-10个样本
'''

from data_prepare.utils import *
from scipy.io import loadmat
import logging
from data_prepare.transform import wavelet_scaterring_analysis, STFT_TF_analysis

# Dataset:['CWRU', 'MFPT', 'IMS', 'PU']  #
# signal_process_method:['original_1d', 'STFT', 'SCWT']  # 支持的信号处理方法
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
def file_list_obtain(mode='Health', number_of_files=20, dir_path=r'D:\F\Data_hub\辛辛那提\IMS\2nd_test'):
    file_name_list = read_file_name_extension_IMS(root_path=dir_path)  # 目录下所有文件名称形成的列表
    if mode == 'Health':
        return file_name_list[51:71]  # 返回从第50个文件开始的20个文件
    else:
        return file_name_list[720:741]  # 返回从第800个文件开始的20个文件


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


def from_index_to_filepath(index=105, root_path=r'D:\F\Data_hub\辛辛那提\IMS'):
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
    result_dict = {os.path.basename(path): path for path in file_path_list}  # 使用字典推导式创建所需的字典
    file_path = result_dict[index]
    return file_path

def filepath_to_samplelist(filepath, sample_len=2048, index=4, label=None, signal_process_method=None, params=None):
    '''
    该函数用来将CWRU数据集中的一个.mat文件的原始振动信号划分为样本，并返回一个由样本组成的列表
    filepath:文件的路径
    sample_len: 单个样本的长度
    label:类别标签
    返回：以列表形式返回
    '''
    data = np.loadtxt(filepath)  # 读取文件内容
    fl = data[:, 0].flatten()
    fl = fl.reshape(-1, )
    data = []
    lab = []
    start, end = 0, sample_len
    while end <= fl.shape[0]:
        x = fl[start:end]
        match signal_process_method:
            case 'original_1d':
                data.append(x)
                lab.append(label)

            case 'STFT':
                processed_tf_img = STFT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                       non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)
            case 'SCWT':
                _, processed_tf_img = wavelet_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                         non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)
            case 'CWT':
                processed_tf_img, _ = CWT_time_frequency(x, fs=params.fs, J=params.J, Q=params.Q,
                                                         non_threshod=params.non_threshod)
                data.append(processed_tf_img)
                lab.append(label)


        # 所提方法处理
        # data.append(x)
        # lab.append(label)
        # 可视化验证
        # if start < 1:
        #     t1 = np.linspace(0, np.max(t), 224)  # 横坐标值缩放
        #     fr1 = np.linspace(0, np.max(fs), 224)  # 纵坐标值缩放
        #     plt.pcolormesh(t1, fr1, np.abs(stft_ZXX_non_resize), shading='gouraud')
        #     plt.title('{}'.format(filename))
        #     plt.ylabel('Frequency [Hz]')
        #     plt.xlabel('Time [sec]')
        #     plt.show()
        start += sample_len
        end += sample_len
    return data, lab

def dataset_construct(files, sample_len=2048, index=4, label=1, signal_process_method='original_1d', SCWT_params=None):
    file_path_list = [from_index_to_filepath(file) for file in files]  # 将文件名称列表转换为路径列表
    # 对每个列表分别处理
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist(file_path, sample_len=sample_len, index=index,
                                           label=label, signal_process_method=signal_process_method, params=SCWT_params)
        samples_list.extend(data)
        labels_list.extend(lab)

    return samples_list, labels_list

class CWT_Params:
    def __init__(self, fs=12e3, J=10, Q=(11, 1), non_threshod=95):
        self.fs = fs
        self.J = int(J)
        self.Q = Q
        self.non_threshod = non_threshod
    def __str__(self):
        return f'CWT_Params(fs={self.fs}, J={self.J}, Q={self.Q}, non_threshod={self.non_threshod})'

class IMS(object):
    def __init__(self,args, sample_len=2560, file_name=None):
        self.sample_len = sample_len
        self.file_name = file_name
        self.args = args

    def single_file_samplelist(self, sample_len=4096):
        file_list = [self.args.data_file_name]
        sample_list, label_list = dataset_construct(files=file_list, sample_len=4096, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        return sample_list, label_list


    def original_1d(self):
        # class1 健康样本标签为1，故障样本为0
        file_list1 = file_list_obtain(mode='Health', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test') # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,20K
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=2560, index=0, label=1, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        file_list2 = file_list_obtain(mode='F', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=2560, index=0, label=0,signal_process_method='original_1d')
        logging.info('number_of_fault_class:{}'.format(len(sample_list2)))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2
        label_list = label_list1 + label_list2
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list
    def STFT(self):
        # class1 健康样本标签为1，故障样本为0
        params = CWT_Params(fs=20e3, J=10, Q=(11, 1), non_threshod=95)
        file_list1 = file_list_obtain(mode='Health', number_of_files=20,
                                      dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,20K
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=0, label=1,
                                                      signal_process_method='STFT', SCWT_params=params)  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        file_list2 = file_list_obtain(mode='F', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=0, label=0,
                                                      signal_process_method='STFT', SCWT_params=params)
        logging.info('number_of_fault_class:{}'.format(len(sample_list2)))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2
        label_list = label_list1 + label_list2
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list

    def CWT(self):
        # class1 健康样本标签为1，故障样本为0
        params = CWT_Params(fs=20e3, J=10, Q=(11, 1), non_threshod=95)
        file_list1 = file_list_obtain(mode='Health', number_of_files=20,
                                      dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,20K
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=0, label=1,
                                                      signal_process_method='CWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        file_list2 = file_list_obtain(mode='F', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=0, label=0,
                                                      signal_process_method='CWT', SCWT_params=params)
        logging.info('number_of_fault_class:{}'.format(len(sample_list2)))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2
        label_list = label_list1 + label_list2
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list

    # def SCWT(self):
    #     # class1 健康样本标签为1，故障样本为0
    #     # class1 健康样本标签为1，故障样本为0
    #     file_list1 = ['97', '98', '100']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
    #     params = CWT_Params(fs=48e3, J=10, Q=(11, 1), non_threshod=95)
    #     logging.info('SCWT_params:{}'.format(params))
    #     sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=4, label=1,
    #                                                   signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
    #     file_list2 = ['99']
    #     sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1,
    #                                                   signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
    #     # 记录日志
    #     logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))
    #
    #     # class2 故障类
    #     file_list3 = ['109', '175', '135', '227']
    #     sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=4, label=0,
    #                                                   signal_process_method='SCWT', SCWT_params=params)
    #     params4 = CWT_Params(fs=12e3, J=10, Q=(11, 1), non_threshod=95)
    #     file_list4 = ['105', '169', '209', '130', '197', '234', '223']
    #     sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=4, label=0,
    #                                                   signal_process_method='SCWT', SCWT_params=params4)
    #     logging.info('number_of_fault_class:{}'.format(len(sample_list3+sample_list4)))
    #     logging.info('测试数据文件:{}/{}/{}/{}'.format(file_list1, file_list2, file_list3, file_list4))
    #     # 将所有的样本组合起来
    #     sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4
    #     label_list = label_list1 + label_list2 + label_list3 + label_list4
    #     logging.info('number_of_samples:{}'.format(len(sample_list)))
    #     return sample_list, label_list
    def SCWT(self):

        # class1 健康样本标签为1，故障样本为0
        file_list1 = file_list_obtain(mode='Health', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test') # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,20K
        params = CWT_Params(fs=20e3, J=10, Q=(11, 1), non_threshod=90)
        logging.info('SCWT_params:{}'.format(params))
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=4, label=1,
                                                      signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))

        # class2 故障类
        params2 = CWT_Params(fs=20e3, J=10, Q=(11, 1), non_threshod=89)
        file_list2 = file_list_obtain(mode='F', number_of_files=20, dir_path=r'E:\datasets\辛辛那提\IMS\2nd_test')
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=4, label=0,
                                                      signal_process_method='SCWT', SCWT_params=params2)

        logging.info('number_of_fault_class:{}'.format(len(sample_list2)))
        logging.info('测试数据文件:{}/{}'.format(file_list1, file_list2))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2
        # sample_list = sample_list3
        # label_list = label_list3
        label_list = label_list1 + label_list2
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list