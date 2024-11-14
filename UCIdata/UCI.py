import os
import pickle
import platform
import sys
from tqdm import tqdm
import numpy as np


class UCIDATA:
    def __init__(self, input_path=r'', read_list=[], output_path=r''):
        self.datas = []
        self.input_path = input_path
        self.read_list = read_list
        self.output_path = output_path
        self.labels = []
        self.filenames = []
        self.data_read()
        self.save_data()


    def data_read(self):
        for root, dirs, files in os.walk(self.input_path):
            for file in tqdm(files):
                file_path = os.path.join(root, file)
                file_list = file_path.split('\\')
                label = int(file_list[-3][1:]) - 1
                if platform.system() == "Windows":
                    px = file_list[-2]
                elif platform.system() == "Linux":
                    px = file_list[-2]

                if file_path.endswith('.txt') and px in self.read_list:
                    data = np.loadtxt(file_path, delimiter=',')
                    data = data.reshape(-1, 5, 9)  # T,V,C
                    self.datas.append(data)
                    self.labels.append(label)
                    self.filenames.append(file_path)

    def save_data(self):
        np.save(self.output_path + '.npy', np.array(self.datas))
        with open('{}_label.pkl'.format(self.output_path), 'wb') as f:
            pickle.dump((self.filenames, self.labels), f)


if __name__ == '__main__':
    train_data = UCIDATA(input_path=r'H:\UCI', read_list=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
                         output_path=r'H:\UCI_PRE_DATA\train')
    test_data = UCIDATA(input_path=r'H:\UCI', read_list=['p8'], output_path=r'H:\UCI_PRE_DATA\test')
