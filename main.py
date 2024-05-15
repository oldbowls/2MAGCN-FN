import pandas as pd
import numpy as np


class Radar_reslove:
    def __init__(self):
        self.data = None
        self.return_data = None
        self.T = None
        # echo、hrrp、rcs、fdpo, direct,latitude_longitude,distance,angle,location
        self.params = ['echo', 'hrrp', 'rcs', 'doppler', 'direction', 'latitude_longitude', 'distance', 'angle',
                       'location']
        self.index_dict = {'echo': [0, 101], 'hrrp': [101, 202], 'rcs': [202, 303], 'doppler': [303, 304],
                           'direction': [304, 307],
                           'latitude_longitude': [307, 310], 'distance': [310, 311], 'angle': [311, 313],
                           'location': [313, 316]}

    def read_data(self, data_path, label_path):
        if 'xlsx' in data_path:
            self.data = np.array(pd.read_excel(data_path, header=None).values.tolist())
            self.label = np.array(pd.read_excel(label_path, header=None).values.tolist())
        elif 'npy' in data_path:
            self.data = np.load(data_path)
            self.label = np.load(label_path)
        else:
            raise 'data or label type is error'

        T, CN = self.data.shape
        self.T = T
        self.data = self.data.reshape((T, CN // 15, 15))

    def combine_data(self, datas):
        datas = np.concatenate(datas, axis=1)
        return datas

    def params_reslover(self, params):  # 主要的解析函数
        return_data = []
        for param in params:
            if param in self.params:
                start_index = self.index_dict[param][0]
                end_index = self.index_dict[param][1]
                return_data.append(self.data[:, start_index: end_index, :])
        return_data = self.combine_data(return_data)  #组合起来
        return return_data, self.label

    def get_detect_sample(self, data, label, index):
        if index >= self.T:
            raise 'The array is out of bounds'
        return data[index, :, :], label, data[index]

    def get_action_sample(self, index, data_length):
        if index + data_length >= self.T:
            raise 'The array is out of bounds'

        start_index = index
        end_index = index + data_length
        #  动作label文件暂时没有。
        return self.return_data[start_index:end_index, :, :]


if __name__ == '__main__':
    data_reslover = Excel_reslove()
    data_reslover.read_data(data_path=r'G:\QTQ\QTQ\data\radar_data\test_data.xlsx')
    # 'echo', 'hrrp', 'rcs', 'doppler', 'direction', 'latitude_longitude', 'distance', 'angle',
    # 'location'
    data = data_reslover.params_reslover(params=['rcs', 'echo', 'hrrp', 'direction', 'latitude_longitude',''])
    print(data.shape)
