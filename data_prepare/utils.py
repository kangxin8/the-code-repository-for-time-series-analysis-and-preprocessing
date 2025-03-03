'''
该脚本主要包含与磁盘交互的文件读取、加载、存储相关的函数类
'''


import os
import glob
import numpy as np
import math
import torch


def expand_array(arr, target_length):
    '''
    该函数用来将一个不等长的2d数组，按照重复采样的方法扩展为一个可进行可视化的2d数组
    Parameters
    ----------
    arr：需要转换的1d数组
    target_length：要扩充的目标长度

    Returns
    -------

    '''
    current_length = len(arr)
    if current_length == target_length:
        return arr
    else:
        # 计算扩展因子
        repeat_factor = target_length // current_length
        remainder = target_length % current_length

        # 每个元素复制repeat_factor次
        expanded_array = np.repeat(arr, repeat_factor)

        # 如果有余数，进一步处理
        if remainder > 0:
            expanded_array = np.concatenate((expanded_array, arr[:remainder]))

        return expanded_array

def read_file_name_extension_IMS(root_path, extension='.json'):
    '''
    该函数用于读取具有指定后缀的文件的绝对路径
    :param root_path:
    :param extension:
    :return:
    '''
    abs_path = os.path.abspath(root_path)
    file_name = []
    for root, dirs, files in os.walk(abs_path):
        for file in files:
            file_name.append(file)
    return file_name

class Sigmoid(object):
    '''
    该函数，包含两个主要的超参数，第一个是指数的基底使得以及
    '''
    def __init__(self, origin_yvalue=0.001, half_xvalue=1.0):
        self.origin_yvalue = origin_yvalue
        self.half_xvalue = half_xvalue

    def __call__(self, x):
        x = torch.tensor(x)
        sigma = math.exp(math.log(1 / self.origin_yvalue - 1) / self.half_xvalue)  # 计算基底值（根据第一和第二个超参数）
        sigma = torch.tensor(sigma)
        x = 1/(1 + torch.pow(sigma, (-x + self.half_xvalue)))  # 确定函数。并对张量进行映射

        return x
# 获取指定路径下所有文件夹的绝对路径，并以列表形式返回
def read_obsolute_path_dir(root_path, skip_dir=None):
    '''
    该函数用来获取指定路径下的所有文件夹的绝对路径，并具有自动跳过指定文件夹的功能
    :param root_path:
    :param skip_dir:
    :return:
    '''
    n_skip_dir = []
    if skip_dir is None:
        skip_dirs = n_skip_dir
    else:
        skip_dirs = n_skip_dir.extend(skip_dir)

    abs_path = os.path.abspath(root_path)  # 获取输入路径的绝对路径
    dir_name_list = os.listdir(abs_path)  # 读取文件夹
    list_dir_obsoute_path = [os.path.join(abs_path, name) for name in dir_name_list
                             if os.path.isdir(os.path.join(abs_path, name)) and name not in skip_dirs]  # 文件夹绝对路径列表
    return list_dir_obsoute_path



# 获取指定文件夹下指定文件的绝对路径，并形成列表
def read_obsolete_path_file(root_path, skip_file=None):
    '''
    该函数用来实现读取指定文件夹下所有文件的绝对路径的能力，且具有跳过指定文件的功能
    :param root_path:
    :param skip_file:
    :return:
    '''
    n_skip_file = []
    if skip_file is None:
        skip_files = n_skip_file
    else:
        skip_files = n_skip_file.extend(skip_file)
    abs_path = os.path.abspath(root_path)
    file_paths = []
    for root, dirs, files in os.walk(abs_path):
        for file in files:
            if file not in skip_files:
                file_paths.append(os.path.join(root, file))
    return file_paths



# 读取指定后缀的文件绝对路径列表
def read_obsolute_path_file_extension(root_path, extension='.mat'):
    '''
    该函数用于读取具有指定后缀的文件的绝对路径
    :param root_path:
    :param extension:
    :return:
    '''
    abs_path = os.path.abspath(root_path)
    file_paths = []
    for root, dirs, files in os.walk(abs_path):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths



# 返回某个文件夹下指定名称前缀的所有文件的个数
def count_files_with_prefix(directory, prefix):
    '''
    该函数用来实现，对指定文件夹下的具有某个前缀的文件的个数
    :param directory: 指定文件夹路径
    :param prefix: 指定前缀
    :return:
    具有指定前缀的文件的个数
    '''
    pattern = os.path.join(directory, prefix + '*')
    files = glob.glob(pattern)
    return len(files)



# 读取指定文件夹下所有文件的名称列表
def read_file_name_extension(root_path, extension='.json'):
    '''
    该函数用于读取具有指定后缀的文件的绝对路径
    :param root_path:
    :param extension:
    :return:
    '''
    abs_path = os.path.abspath(root_path)
    file_name = []
    for root, dirs, files in os.walk(abs_path):
        for file in files:
            if file.endswith(extension):
                file_name.append(file)
    return file_name


# 对文件名称进行修改，将文件所在文件夹和文件夹所在文件夹的名称作为前缀
def add_parent_folder_names_as_prefix(directory):
    '''
    Parameters
    ----------
    directory:文件所在的文件夹路径
    Returns
    -------
    '''
    # 遍历目录中的所有文件，对名称进行逐个的修改
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)  # 获取当前文件的完整路径
            # =====创建新的完整路径======
            parent_folder = os.path.basename(root)  # 当前文件夹名称
            grandparent_folder = os.path.basename(os.path.dirname(root))  # 上级文件夹名称
            new_file_name = f"{grandparent_folder}_{parent_folder}_{file}"  # 新的文件名
            new_file_path = os.path.join(root, new_file_name)  # 完整的路径

            os.rename(file_path, new_file_path)  # 重命名文件
            print(f"Renamed: {file_path} -> {new_file_path}")
    pass

# 对文件名称进行修改，将文件所在文件夹名称作为前缀
def add_folder_names_as_prefix(directory):
    '''
    Parameters
    ----------
    directory:文件所在的文件夹路径
    Returns
    -------
    '''
    # 遍历目录中的所有文件，对名称进行逐个的修改
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)  # 获取当前文件的完整路径
            # =====创建新的完整路径======
            parent_folder = os.path.basename(root)  # 当前文件夹名称
            new_file_name = f"{parent_folder}_{file}"  # 新的文件名
            new_file_path = os.path.join(root, new_file_name)  # 完整的路径

            os.rename(file_path, new_file_path)  # 重命名文件
            print(f"Renamed: {file_path} -> {new_file_path}")
    pass

if __name__ == '__main__':
    # ====修改文件名称=======
    # ======获取所有需要更改的文件夹绝对路径======
    root = r'E:\datasets\PHM\PHM2024\Preliminary_Stage\Datasets\Training\训练集'  # 上级文件夹路径
    parent_dir_list = read_obsolute_path_dir(root)  # 该文件夹下所有的文件夹的绝对路径
    for dir in parent_dir_list:
        dir_list = read_obsolute_path_dir(dir)
        for sub_dir in dir_list:
            add_parent_folder_names_as_prefix(sub_dir)





