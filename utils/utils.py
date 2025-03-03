'''
算法实现的一些辅助函数
'''
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
