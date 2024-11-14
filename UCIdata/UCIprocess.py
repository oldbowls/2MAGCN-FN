import cv2
from numpy import random as nprand
import random
# import imutils
import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import warnings
import math
from dataset.rotation import *

def uniform_sample_np(data_numpy, size):
    T, V, C = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[uniform_list]

def center(data):  # 所以一定要注意，节点的顺序必须是固定的
    T, V, C = data.shape
    data[:, :, :] -= data[0:1, 0:1, :]
    return data

def norm(datas):
    t, v, c = datas.shape
    # 将数据重塑为二维数组，其中每一列对应一个通道
    reshaped_data = datas.reshape((t * v, c))

    # 寻找每个通道的最大值和最小值
    max_values = np.max(reshaped_data, axis=0)
    min_values = np.min(reshaped_data, axis=0)
    # max_values = np.array([180, 90, 180])
    # min_values = np.array([-180, -90, -180])
    # 将每个通道的数据进行归一化处理
    normalized_data = (reshaped_data - min_values) / (max_values - min_values)
    # 将数据还原为原始形状
    datas = normalized_data.reshape((t, v, c))
    return datas