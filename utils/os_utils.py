'''
该脚本主要包含与磁盘交互的文件读取、加载、存储相关的函数类
'''


import os
import glob


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



