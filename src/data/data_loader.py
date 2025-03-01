import os
import numpy as np
import pandas as pd
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from collections import defaultdict

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', 
                 cols=None, online = 'full'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        self.df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        
        self.len_df = len(self.df_raw)
        
        
        mapping = {
                ('ETTh1', 'ETTh2'): {
                    'start':{
                        'train': 0,
                        'val': 4*30*24 - self.seq_len,
                        'test': 5*30*24 - self.seq_len
                    },
                    'end':{
                        'train': 4*30*24,
                        'val': 5*30*24,
                        'test': 20*30*24
                    }
                },
                ('ETTm1', 'ETTm2'): {
                    'start':{
                        'train': 0,
                        'val': 4*30*24*4 - self.seq_len,
                        'test': 5*30*24*4 - self.seq_len
                    },
                    'end':{
                        'train': 4*30*24*4,
                        'val': 5*30*24*4,
                        'test': 20*30*24*4
                    }
                },
                ('Electricity', 'Exchange', 'ILI', 'Weather', 'WTH', 'Traffic'): {
                    'start': {
                        'train': 0,
                        'val': int(self.len_df*0.2),
                        'test': int(self.len_df*0.25),                
                    },
                    'end': {
                        'train': int(self.len_df*0.2),
                        'val': int(self.len_df*0.25),
                        'test': int(self.len_df)
                    }
                }
        }
        
        self.borders = {}
        for k, v in mapping.items():
            for key in k: self.borders[key] = v
        self.borders = self.borders[data]
        
        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1 = self.borders['start'][self.flag]
        border2 = self.borders['end'][self.flag]
        
        # TODO: refactor this
                
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        # import ipdb; ipdb.set_trace()

        if self.scale:
            train_data = df_data[self.borders['start']['train']:self.borders['end']['train']]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.timeenc == 2:
            train_df_stamp = df_raw[['date']][self.borders['start']['train']:self.borders['end']['train']]
            train_df_stamp['date'] = pd.to_datetime(train_df_stamp.date)
            train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
            date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            data_stamp = date_scaler.transform(data_stamp)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
