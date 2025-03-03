'''
该脚本用于将基于CWRU构建的数据集以样本和标签的列表形式返回
关于CWRU数据的一些关键信息
1、取用风扇端传感器测得的数据，index=4;驱动端传感器测得的数据，index=3;99.mat正常数据，需要设置index=6
'''

from data_prepare.utils import *
from scipy.io import loadmat
import logging
from data_prepare.transform import wavelet_scaterring_analysis, STFT_TF_analysis

# Dataset:['CWRU', 'MFPT', 'IMS', 'PU']  #
# signal_process_method:['original_1d', 'STFT', 'SCWT']  # 支持的信号处理方法

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

def from_index_to_filepath(index=105, root_path=r'D:\F\Data_hub\凯斯西储大学数据'):
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
    desired_key = data_key_list[args.CWRU_dic_index]  # 指定键的名称
    fl = data[desired_key].flatten()
    fl = fl.reshape(-1,)
    data = []
    lab = []
    start, end = 0, args.sample_len
    while end <= fl.shape[0]:
        x = fl[start:end]
        match signal_process_method:
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
                # f, t, imgs = stft_window(x, fs=fs, nperseg=nperseg, noverlap=noverlop)  # 按照指定的超参数做STFT变换，返回绝对值
                # quartiles = np.percentile(imgs.flatten(), [25, 50, threshhold])  # 四分位数
                # origin_yvalue = 0.001  # 超参数1的选择
                # half_xvalue = quartiles[2] * 100.0  # 超参数2的选择
                # f = Sigmoid(origin_yvalue, half_xvalue)  # 实例化非线性函数
                # ZXX_non = imgs * 100.0  # 标准化时频图,非线性函数的输入
                # stft_ZXX_non = f(ZXX_non)  # 非线性处理
                # stft_ZXX_non_resize = resize(imgs, output_shape=(224, 224), anti_aliasing=True)  # 尺寸校准


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

def dataset_construct(args, files=None, sample_len=2048, index=4, label=1,file_info=None, signal_process_method='original_1d', SCWT_params=None):
    if file_info is None:
        file_list = [args.data_file_name]
    else:
        file_list = file_info

    file_path_list = [from_index_to_filepath(file) for file in file_list]  # 将文件名称列表转换为路径列表
    # 对每个列表分别处理
    samples_list = []  # 样本列表
    labels_list = []  # 标签列表
    for file_path in file_path_list:
        data, lab = filepath_to_samplelist(args, file_path, sample_len=args.sample_len, index=args.CWRU_dic_index,
                                           label=label, signal_process_method=signal_process_method, params=SCWT_params)
        samples_list.extend(data)
        labels_list.extend(lab)

    return samples_list, labels_list

class preprocess_Params:
    def __init__(self, sam_len=None, fs=48e3, fr=1797, J=10, Q=(11, 1), nperseg=256, noverlap=250, non_threshod=95):
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
class CWRU(object):
    def __init__(self, args):
        self.sample_len = args.sample_len
        self.args = args
        preprocess_params_dic = {'dic_index':3, 'label':0,  # 样本参数
            'sam_len':self.sample_len, 'fs':48e3, 'fr':1797,
                                  'J':10, 'Q':(11, 1),  # 小波变换参数
                                  'nperseg':256, 'noverlap':250,  # STFT参数
                                  'non_threshod':95}
        self.preprocess_params = DictObj(preprocess_params_dic)


    def single_file_samplelist(self):
        sample_list, label_list = dataset_construct(self.args, self.preprocess_params, sample_len=4096, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        return sample_list, label_list
    def single_file_sample_obtain(self, sample_index=5):
        '''该函数用来获取指定的单个样本'''
        file_list = [self.file_name]
        sample_list, label_list = dataset_construct(files=file_list, sample_len=self.sample_len, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        sample = sample_list[sample_index]
        return sample
    def original_1d(self):
        # region健康、故障二分类
        # class1 健康样本标签为0，故障样本为1
        # file_list1 = ['97', '98', '100']   # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        # sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=4096, index=3, label=0, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # file_list2 = ['99']
        # sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=4096, index=6, label=0, signal_process_method='original_1d')  # 将列表中的文件转换为样本
        # logging.info('number_of_normal_class:{}'.format(len(sample_list1+sample_list2)))  # 记录日志
        #
        # # class2 故障类
        # file_list3 = ['105', '169', '209', '130', '197', '234', '223', '109', '175', '135']
        # sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=4096, index=3, label=0,signal_process_method='original_1d')
        # logging.info('number_of_fault_class:{}'.format(len(sample_list3)))  # 记录日志
        #
        # # 将所有的样本组合起来
        # sample_list = sample_list1 + sample_list2 + sample_list3
        # label_list = label_list1 + label_list2 + label_list3
        # logging.info('number_of_samples:{}'.format(len(sample_list)))
        # endregion
        # region 三分类（健康、内圈、外圈） 健康样本标签为0，内圈故障标签为1，外圈故障样本标签为2
        # 健康样本 0
        file_list1 = ['97', '98', '100']  # 要读取的同类文件（文件具有相同的类型，相同的采样频率）,48K
        sample_list1, label_list1 = dataset_construct(files=file_list1, sample_len=4096, index=3, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        file_list2 = ['99']
        sample_list2, label_list2 = dataset_construct(files=file_list2, sample_len=4096, index=6, label=0,
                                                      signal_process_method='original_1d')  # 将列表中的文件转换为样本
        logging.info('number_of_normal_class:{}'.format(len(sample_list1 + sample_list2)))  # 记录日志
        # 内圈故障 1
        file_list3 = ['105', '106', '169', '170', '209', '210']  # 12K驱动端轴承内圈故障
        sample_list3, label_list3 = dataset_construct(files=file_list3, sample_len=4096, index=3, label=1,
                                                      signal_process_method='original_1d')
        file_list4 = ['109', '110', '174', '175', '213', '214']  # 48K驱动端轴承内圈故障
        sample_list4, label_list4 = dataset_construct(files=file_list4, sample_len=4096, index=3, label=1,
                                                      signal_process_method='original_1d')
        file_list5 = ['278', '279', '274', '275', '270', '271']  # 12K风扇端轴承内圈故障
        sample_list5, label_list5 = dataset_construct(files=file_list5, sample_len=4096, index=4, label=1,
                                                      signal_process_method='original_1d')
        logging.info('number_of_IF_class:{}'.format(len(sample_list3 + sample_list4 + sample_list5)))  # 记录日志

        # 外圈故障 2
        file_list6 = ['130', '131', '234', '235', '144', '145']  # 12K驱动端轴承外圈故障
        sample_list6, label_list6 = dataset_construct(files=file_list6, sample_len=4096, index=3, label=2,
                                                      signal_process_method='original_1d')
        file_list7 = ['135', '136', '201', '202', '238', '239', '148', '149']  # 48K驱动端轴承外圈故障
        sample_list7, label_list7 = dataset_construct(files=file_list7, sample_len=4096, index=3, label=2,
                                                      signal_process_method='original_1d')
        file_list8 = ['294', '295', '313', '315', '309', '310']  # 12K风扇端轴承外圈故障
        sample_list8, label_list8 = dataset_construct(files=file_list8, sample_len=4096, index=4, label=2,
                                                      signal_process_method='original_1d')
        logging.info('number_of_OF_class:{}'.format(len(sample_list6 + sample_list7 + sample_list8)))  # 记录日志

        # 将所有的样本组合起来
        sample_list = sample_list1 + sample_list2 + sample_list3 + sample_list4 + sample_list5 + sample_list6 + sample_list7 + sample_list8
        label_list = label_list1 + label_list2 + label_list3 + label_list4 + label_list5 + label_list6 + label_list7 + label_list8
        logging.info('number_of_total_samples:{}'.format(len(sample_list)))
        # endregion
        return sample_list, label_list
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