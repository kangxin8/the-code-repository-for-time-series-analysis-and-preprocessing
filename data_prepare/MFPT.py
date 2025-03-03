'''
该脚本用于将基于MFPT构建的数据集以样本和标签的列表形式返回
关于MFPT数据的一些关键信息
1、采样频率有两个：97656，48828
'''

from data_prepare.utils import *
from scipy.io import loadmat
import logging
from data_prepare.transform import wavelet_scaterring_analysis, STFT_TF_analysis
# from signal_process.signal_1d_transform_and_analysis import angular_resample
from scipy.signal import hilbert
# Dataset:['CWRU', 'MFPT', 'IMS', 'PU']  #
# signal_process_method:['original_1d', 'STFT', 'SCWT']  # 支持的信号处理方法
def normalization_processing(data):
    data_mean = data.mean()
    data_std = data.std()

    data = data - data_mean
    data = data / data_std

    return data
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
    STFT_analysis = STFT_TF_analysis(fs=fs, nperseg=512, noverlap=500)  # 实例化小波散射对象
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


def from_index_to_filepath(index=105, root_path=r'E:\datasets\MFPT Fault Data Sets'):
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
        file_list = read_obsolute_path_file_extension(dir, extension='.mat')  # 获取文件夹下所有指定后缀文件的绝对路径
        file_path_list.extend(file_list)
    result_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in file_path_list}  # 使用字典推导式创建所需的字典
    file_path = result_dict[index]
    return file_path

def filepath_to_samplelist(args, filepath, sample_len=2048, index=4, label=None, signal_process_method=None, params=None):
    '''
    该函数用来将CWRU数据集中的一个.mat文件的原始振动信号划分为样本，并返回一个由样本组成的列表
    filepath:文件的路径
    sample_len: 单个样本的长度
    label:类别标签
    返回：以列表形式返回
    '''
    data = loadmat(filepath)  # 读取文件内容
    data_key_list = list(data.keys())  # 读取出的内容以字典形式存储,获取键的列表
    desired_key = data_key_list[args.MFPT_dic_index]  # 指定键的名称
    desired_dic = data[desired_key]
    fl = np.array(desired_dic[0]['gs'].tolist()).flatten()
    fl = fl.reshape(-1, )
    data = []
    lab = []
    start, end = 0, args.sample_len
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
def filepath_to_samplelist_plus(filepath, sample_len=2048, index=4, label=None, sam_fre=None, signal_process_method=None, params=None):
    '''
    该函数用来将CWRU数据集中的一个.mat文件的原始振动信号划分为样本，并返回一个由样本组成的列表
    filepath:文件的路径
    sample_len: 单个样本的长度
    label:类别标签
    返回：以列表形式返回
    '''
    RPM = 25  # 获取转速信息
    unified_Os = 350  # 每转的采样个数
    # 数据读取
    data = loadmat(filepath)  # 读取文件内容
    data_key_list = list(data.keys())  # 读取出的内容以字典形式存储,获取键的列表
    desired_key = data_key_list[index]  # 指定键的名称
    desired_dic = data[desired_key]
    fl = np.array(desired_dic[0]['gs'].tolist()).flatten()
    fl = fl.reshape(-1, )
    # 角域重采样
    rotating_speed = np.ones(len(fl)) * RPM
    this_ctgr_data = angular_resample(fl, rotating_speed, sam_fre, unified_Os)
    data = []
    lab = []
    start, end = 0, sample_len
    while end <= this_ctgr_data.shape[0]:
        x = this_ctgr_data[start:end]
        x = normalization_processing(x)  # 首先进行正则化
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
            case 'envelop':
                sample = np.abs(hilbert(x)).astype(np.float32)

                data.append(sample)
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

def angular_dataset_construct(files, sample_len=2048, index=4, label=1, sam_fre=None, signal_process_method='original_1d', SCWT_params=None):
    file_path_list = [from_index_to_filepath(file) for file in files]  # 将文件名称列表转换为路径列表
    # 对每个列表分别处理
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist_plus(file_path, sample_len=sample_len, index=index, sam_fre=sam_fre,
                                           label=label, signal_process_method=signal_process_method, params=SCWT_params)
        samples_list.extend(data)
        labels_list.extend(lab)

    return samples_list, labels_list
def dataset_construct(args, files=None, sample_len=2048, index=4, label=1, signal_process_method='original_1d', SCWT_params=None, file_info=None):
    if file_info is None:
        file_list = [args.data_file_name]
    else:
        file_list = file_info

    file_path_list = [from_index_to_filepath(file) for file in file_list]  # 将文件名称列表转换为路径列表
    # 对每个列表分别处理
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist(args, file_path, sample_len=args.sample_len, index=args.MFPT_dic_index,
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
class MFPT(object):
    def __init__(self, args, domain_name=None, sample_len=1024, preprocess_name=None, file_name='97', single_domain=True, sub_dataset_name='a01', preprocess='STFT_non'):
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
    # 获取指定文件形成的列表
    def single_file_samplelist(self):
        sample_list, label_list = dataset_construct(self.args, sample_len=4096,
                                                    index=3, label=0, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        return sample_list, label_list
    # 获取指定文件的单个样本
    def single_file_sample_obtain(self, sample_index=5):
        file_list = [self.file_name]
        sample_list, label_list = dataset_construct(files=file_list, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        sample = sample_list[sample_index]
        return sample
    def angular_resample_1d(self):
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['baseline_2', 'baseline_3']   # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,97656
        sample_list1, label_list1 = angular_dataset_construct(files=file_list1, sample_len=self.sample_len,
                                                              index=3, label=0, sam_fre=97656, signal_process_method='envelop')  # 将列表中的文件转换为样本

        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        file_list3 = ['InnerRaceFault_vload_1']
        sample_list3, label_list3 = angular_dataset_construct(files=file_list3, sample_len=self.sample_len,
                                                              index=3, label=1, sam_fre=48828, signal_process_method='envelop')
        file_list4 = ['OuterRaceFault_2']
        sample_list4, label_list4 = angular_dataset_construct(files=file_list4, sample_len=self.sample_len,
                                                              index=3, label=1, sam_fre=97656, signal_process_method='envelop')

        logging.info('number_of_fault_class:{}'.format(len(sample_list3+sample_list4)))

        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list3 + sample_list4
        label_list = label_list1 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list
    def original_1d(self):
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['baseline_2', 'baseline_3']   # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,97656
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=3, label=0, signal_process_method='original_1d')  # 将列表中的文件转换为样本

        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        file_list3 = ['InnerRaceFault_vload_1']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=3, label=1, signal_process_method='original_1d')
        file_list4 = ['OuterRaceFault_2']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=3, label=1, signal_process_method='original_1d')

        logging.info('number_of_fault_class:{}'.format(len(sample_list3+sample_list4)))

        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list3 + sample_list4
        label_list = label_list1 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list
    def STFT(self):
        file_list1 = ['baseline_2']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,97656
        params1 = CWT_Params(fs=97656, J=10, Q=(11, 1), non_threshod=95)
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=3, label=1,
                                                      signal_process_method='STFT', SCWT_params=params1)  # 将列表中的文件转换为样本

        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        params2 = CWT_Params(fs=48828, J=10, Q=(11, 1), non_threshod=95)

        file_list3 = ['InnerRaceFault_vload_1']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='STFT', SCWT_params=params2)
        file_list4 = ['OuterRaceFault_2']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='STFT', SCWT_params=params1)

        logging.info('number_of_fault_class:{}'.format(len(sample_list3 + sample_list4)))

        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list3 + sample_list4
        label_list = label_list1 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list

    def CWT(self):
        file_list1 = ['baseline_2']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,97656
        params1 = CWT_Params(fs=97656, J=10, Q=(11, 1), non_threshod=95)
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=3, label=1,
                                                      signal_process_method='CWT', SCWT_params=params1)  # 将列表中的文件转换为样本

        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        params2 = CWT_Params(fs=48828, J=10, Q=(11, 1), non_threshod=95)

        file_list3 = ['InnerRaceFault_vload_1']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='CWT', SCWT_params=params2)
        file_list4 = ['OuterRaceFault_2']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='CWT', SCWT_params=params1)

        logging.info('number_of_fault_class:{}'.format(len(sample_list3 + sample_list4)))

        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list3 + sample_list4
        label_list = label_list1 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list

    # 该函数专门为了第一篇小论文返修时，为了测试MFPT数据所编写的包含b01-b15十五个任务.
    # 通过传入单域和子域名称参数，得到不同的输出结果
    def first_revised_stft_non(self):
        params = CWT_Params(fs=48e3, J=10, Q=(11, 1), nperseg=256, noverlap=250, non_threshod=95)

        a01 = ['baseline_1', 'baseline_2', 'baseline_3']
        a02 = ['OuterRaceFault_vload_1']
        a03 = ['OuterRaceFault_vload_2']
        a04 = ['OuterRaceFault_vload_3']
        a05 = ['OuterRaceFault_vload_4']
        a06 = ['OuterRaceFault_vload_5']
        a07 = ['OuterRaceFault_vload_6']
        a08 = ['OuterRaceFault_vload_7']
        a09 = ['InnerRaceFault_vload_1']
        a10 = ['InnerRaceFault_vload_2']
        a11 = ['InnerRaceFault_vload_3']
        a12 = ['InnerRaceFault_vload_4']
        a13 = ['InnerRaceFault_vload_5']
        a14 = ['InnerRaceFault_vload_6']
        a15 = ['InnerRaceFault_vload_7']

        subdata_list = [a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15]
        if self.single_domain:
            # region 单一子数据集获取
            match self.sub_dataset_name:
                case 'a01':
                    sample_list, label_list = dataset_construct(files=a01, sample_len=self.sample_len, index=3,
                                                                label=0, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a01, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a02':
                    sample_list, label_list = dataset_construct(files=a02, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a02, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a03':
                    sample_list, label_list = dataset_construct(files=a03, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a03, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a04':
                    sample_list, label_list = dataset_construct(files=a04, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a04, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a05':
                    sample_list, label_list = dataset_construct(files=a05, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a05, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a06':
                    sample_list, label_list = dataset_construct(files=a06, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a06, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a07':
                    sample_list, label_list = dataset_construct(files=a07, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a07, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a08':
                    sample_list, label_list = dataset_construct(files=a08, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a08, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a09':
                    sample_list, label_list = dataset_construct(files=a09, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a09, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a10':
                    sample_list, label_list = dataset_construct(files=a10, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a10, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a11':
                    sample_list, label_list = dataset_construct(files=a11, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a11, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a12':
                    sample_list, label_list = dataset_construct(files=a12, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a12, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a13':
                    sample_list, label_list = dataset_construct(files=a13, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a13, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a14':
                    sample_list, label_list = dataset_construct(files=a14, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a14, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
                case 'a15':
                    sample_list, label_list = dataset_construct(files=a15, sample_len=self.sample_len, index=3,
                                                                label=1, signal_process_method=self.preprocess,
                                                                SCWT_params=params)  # 将列表中的文件转换为样本
                    logging.info(
                        'test_dataset_info:\nsample_list={},index=4, sam_len=4096\nstft_param={},{},'
                        'non_threshod={}'
                        .format(a15, params.nperseg, params.noverlap, params.non_threshod)
                    )
                    return sample_list, label_list
            # endregion

        else:
            Sample_list, Label_list = [], []
            for i, data_list in enumerate(subdata_list):
                if i==0:
                    label = 0
                else:
                    label = 1
                sample_list, label_list = dataset_construct(files=data_list, sample_len=self.sample_len, index=3,
                                                            label=label, signal_process_method=self.preprocess,
                                                            SCWT_params=params)
                Sample_list = Sample_list + sample_list
                Label_list = Label_list + label_list

            return Sample_list, Label_list
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
        # class1 健康样本标签为1，故障样本为0
        file_list1 = ['baseline_2']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        params = CWT_Params(fs=97656, J=11, Q=(11, 1), non_threshod=92)
        logging.info('SCWT_params:{}'.format(params))
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=self.sample_len, index=3, label=1,
                                                      signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=self.sample_len, index=7, label=1,
        #                                               signal_process_method='SCWT', SCWT_params=params)  # 将列表中的文件转换为样本
        # # 记录日志
        logging.info('number_of_normal_class:{}'.format(len(sample_list1)))
        # class2 故障类
        params3 = CWT_Params(fs=48828, J=11, Q=(11, 1), non_threshod=92)
        file_list3 = ['InnerRaceFault_vload_1']
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='SCWT', SCWT_params=params3)
        params4 = CWT_Params(fs=97656, J=11, Q=(11, 1), non_threshod=92)
        file_list4 = ['OuterRaceFault_2']
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='SCWT', SCWT_params=params4)
        logging.info('number_of_fault_class:{}'.format(len(sample_list3+sample_list4)))
        logging.info('测试数据文件:{}/{}/{}'.format(file_list1, file_list3, file_list4))
        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list3 + sample_list4
        # sample_list = sample_list3
        # label_list = label_list3
        label_list = label_list1 + label_list3 + label_list4
        logging.info('number_of_samples:{}'.format(len(sample_list)))
        return sample_list, label_list



if __name__ == '__main__':
    Dataset = MFPT(sample_len=4096)
    sample_list, label_list = Dataset.first_revised_stft_non()