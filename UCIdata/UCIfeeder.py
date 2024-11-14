import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .UCIprocess import uniform_sample_np, center, norm


class DailySportsDataset(Dataset):
    def __init__(self, data_path, window_size=10, mode='train'):
        """
        :param data_path: 数据文件夹的路径
        :param window_size: 滑动时间窗大小
        :param step_size: 滑动步长
        :param mode: 'train' 或 'test'
        """

        self.window_size = window_size
        self.data_path = data_path
        self.mode = mode
        self.read_data()

    def read_data(self):
        file_path = os.path.join(self.data_path, self.mode + '.npy')
        self.datas = np.load(file_path)

        label_path = os.path.join(self.data_path, '{}_label.pkl'.format(self.mode))
        try:
            with open(label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

    def precess(self):
        pass

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, index):

        data = self.datas[index]
        sample_name = self.sample_name[index]
        label = self.label[index]
        data = uniform_sample_np(data, self.window_size)

        data = center(data)
        # data = norm(data)
        return data, label, sample_name

    def top_k(self, score, top_k):
        print(self.label)
        rank = score.argsort()  # 从小到大排序，且返回的是下标
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]  # 找到了每一个样本中分数最高的那个,并判断是否和label相同
        return sum(hit_top_k) * 1.0 / len(hit_top_k)  # 返回准确率


if __name__ == "__main__":
    dataset = DailySportsDataset(r'H:\UCI_PRE_DATA', mode='train')
