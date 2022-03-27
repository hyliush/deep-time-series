import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from fastNLP import cache_results
from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')
class DatasetBase(Dataset):
    def __init__(self, data_path, size, features, file_name, 
                 target, scale, inverse, timeenc, freq, cols):
        self.data_path = data_path
        self.size = size
        self.features = features
        self.file_name = file_name
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

class Dataset_ETT_hour(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):

        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        self.train_idxs = np.arange(border1s[0], border2s[0]-self.seq_len)
        self.val_idxs = np.arange(border1s[1], border2s[1]-self.seq_len)
        self.test_idxs = np.arange(border1s[2], border2s[2]-self.seq_len)

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # scale处理后为data，未处理为df_data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            # 历史label_len 和 真实需要预测的长度pred_len
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

class Dataset_ETT_minute(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):

        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        self.train_idxs = np.arange(border1s[0], border2s[0]-self.seq_len)
        self.val_idxs = np.arange(border1s[1], border2s[1]-self.seq_len)
        self.test_idxs = np.arange(border1s[2], border2s[2]-self.seq_len)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
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
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path,
                                          self.file_name))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        self.train_idxs = np.arange(border1s[0], border2s[0]-self.seq_len)
        self.val_idxs = np.arange(border1s[1], border2s[1]-self.seq_len)
        self.test_idxs = np.arange(border1s[2], border2s[2]-self.seq_len)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
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

class Dataset_Pred(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path,
                                          self.file_name))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

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
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class VolatilityDataSetSeq2Seq(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='d', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.test_size = 60
        self.seq_len, self.label_len, self.pred_len = size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df_raw = df_raw.drop(columns=["stock_id", "target", "weekday", "time_id", "holiday_name", "holiday_tag", "holiday_tag_cumsum"])
        df_raw = df_raw.rename(columns={"Date":"date"})

        total_len, start_point = len(df_raw), 0
        length = total_len-self.seq_len-self.pred_len+1
        cut_point1, cut_point2 = length-3*self.test_size, length-2*self.test_size
        border1s = [start_point, cut_point1, cut_point2]
        border2s = [cut_point1, cut_point2, length]
        self.train_idxs = np.arange(border1s[0], border2s[0])
        self.val_idxs = np.arange(border1s[1], border2s[1])
        self.test_idxs = np.arange(border1s[2], border2s[2])

        if isinstance(self.features, str):
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            # encoder输入data_x,保持scaler，decoder输入data_y
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


class VolatilityDataSetNoraml(DatasetBase):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0,
                  freq='d', cols=None, horizon = 0):
        '''horizon: predict timeseries from horizon+1 to horizon+1+pred in head. default(0) '''
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.test_size = 60
        self.seq_len, self.label_len, self.pred_len = size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # @cache_results(_cache_fp=None)
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df_raw = df_raw.drop(columns=["stock_id", "target", "weekday", "time_id", "holiday_name", "holiday_tag", "holiday_tag_cumsum"])
        df_raw = df_raw.rename(columns={"Date":"date"})

        length = len(df_raw)-self.seq_len-self.pred_len+1
        cut_point1, cut_point2 = length-3*self.test_size, length-2*self.test_size
        border1s = [0, cut_point1, cut_point2]
        border2s = [cut_point1, cut_point2, length]
        self.train_idxs = np.arange(border1s[0], border2s[0])
        self.val_idxs = np.arange(border1s[1], border2s[1])
        self.test_idxs = np.arange(border1s[2], border2s[2])

        if isinstance(self.features, str):
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.horizon 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y#, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class VolatilityDataSetGate(Dataset):
    def __init__(self, data_path, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='d', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.test_size = 60
        self.seq_len, self.label_len, self.pred_len = size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
        length = len(df_raw)-self.seq_len-self.pred_len+1
        cut_point1, cut_point2 = length-3*self.test_size, length-2*self.test_size
        border1s = [0, cut_point1, cut_point2]
        border2s = [cut_point1, cut_point2, length]
        self.train_idxs = np.arange(border1s[0], border2s[0])
        self.val_idxs = np.arange(border1s[1], border2s[1])
        self.test_idxs = np.arange(border1s[2], border2s[2])

        # spatial
        self.data_spatial = df_raw[["stock_id"]].values
        
        df_raw = df_raw.drop(columns=["stock_id", "target", "weekday", "time_id", "holiday_name", "holiday_tag", "holiday_tag_cumsum"])
        df_raw = df_raw.rename(columns={"Date":"date"})
        
        if isinstance(self.features, list):
            df_data = df_raw[self.features]
        if isinstance(self.features, str):
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # temporal
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_stamp = data_stamp
        
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
    
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
        x_temporal = self.data_stamp[s_begin:s_end]
        x_spatial = self.data_spatial[s_begin:s_end]

        return seq_x, x_temporal, x_spatial, seq_y#, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
import json
from typing import Optional
class OzeDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 labels_path: Optional[str] = "labels.json",
                 normalize: Optional[str] = "max",
                 **kwargs):
        """Load dataset from npz."""
        super().__init__(**kwargs)

        self._normalize = normalize

        self._load_npz(dataset_path, labels_path)

    def _load_npz(self, dataset_path, labels_path):
        # Load dataset as csv
        dataset = np.load(dataset_path)

        # Load labels, can be found through csv or challenge description
        with open(labels_path, "r") as stream_json:
            self.labels = json.load(stream_json)

        R = dataset['R'].astype(np.float32)
        X = dataset['X'].astype(np.float32).transpose(0, 2, 1)
        Z = dataset['Z'].astype(np.float32).transpose(0, 2, 1)

        m = Z.shape[0]  # Number of training example
        K = Z.shape[1]  # Time serie length

        R = np.tile(R[:, np.newaxis, :], (1, K, 1))

        # Store R, Z and X as x and y
        self._x = np.concatenate([Z, R], axis=-1)
        self._y = X

        # Normalize
        if self._normalize == "mean":
            mean = np.mean(self._x, axis=(0, 1))
            std = np.std(self._x, axis=(0, 1))
            self._x = (self._x - mean) / (std + np.finfo(float).eps)

            self._mean = np.mean(self._y, axis=(0, 1))
            self._std = np.std(self._y, axis=(0, 1))
            self._y = (self._y - self._mean) / (self._std + np.finfo(float).eps)
        elif self._normalize == "max":
            M = np.max(self._x, axis=(0, 1))
            m = np.min(self._x, axis=(0, 1))
            self._x = (self._x - m) / (M - m + np.finfo(float).eps)

            self._M = np.max(self._y, axis=(0, 1))
            self._m = np.min(self._y, axis=(0, 1))
            self._y = (self._y - self._m) / (self._M - self._m + np.finfo(float).eps)
        elif self._normalize is None:
            pass
        else:
            raise(
                NameError(f'Normalize method "{self._normalize}" not understood.'))

        # Convert to float32
        self._x = torch.Tensor(self._x)
        self._y = torch.Tensor(self._y)

    def inverse_transform(self,
                y: np.ndarray,
                idx_label: int) -> torch.Tensor:
        if self._normalize == "max":
            return y * (self._M[idx_label] - self._m[idx_label] + np.finfo(float).eps) + self._m[idx_label]
        elif self._normalize == "mean":
            return y * (self._std[idx_label] + np.finfo(float).eps) + self._mean[idx_label]
        else:
            raise(
                NameError(f'Normalize method "{self._normalize}" not understood.'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._x[idx], self._y[idx]
    def __len__(self):
        return self._x.shape[0]

class UbiquantInformer(Dataset):
    def __init__(self, data_path, flag, size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='d', cols=None, test_size=60):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.test_size = 30
        if size == None:
            self.seq_len = 20
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path,
                                          self.file_name))
        df_raw = df_raw.drop(columns=["row_id", "time_id", "investment_id"])
        length = len(df_raw) - 1
        cut_point1, cut_point2 = length-3*self.test_size, length-self.test_size
        
        border1s = [0, cut_point1 - self.seq_len, cut_point2 - self.seq_len]
        border2s = [cut_point1, cut_point2, length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(self.features, list):
            df_data = df_raw[self.features]
        if isinstance(self.features, str):
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        data_stamp = df_raw[['time_id']][border1:border2].values

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
            # encoder输入data_x,保持scaler，decoder输入data_y
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

class UbiquantDataSetNoraml(Dataset):
    def __init__(self, data_path, flag='train', size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0,
                  freq='d', cols=None, test_size=60, horizon = 0):
        '''horizon: predict timeseries from horizon+1 to horizon+1+pred in head. default(0) '''
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.test_size = 30
        if size == None:
            self.seq_len = 20
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path,
                                          self.file_name))
        df_raw["target"] = df_raw["target"].shift()
        df_raw = df_raw.dropna()
        df_raw = df_raw.drop(columns=["row_id", "time_id", "investment_id"])
        length = len(df_raw) - 1
        cut_point1, cut_point2 = length-2*self.test_size, length-self.test_size
        
        border1s = [0, cut_point1 - self.seq_len, cut_point2 - self.seq_len]
        border2s = [cut_point1, cut_point2, length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(self.features, list):
            df_data = df_raw[self.features]
        if isinstance(self.features, str):
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.horizon 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y,# seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class ToyDataset(DatasetBase):
    def __init__(self,data_path="", flag='train', size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.seq_len, self.label_len, self.pred_len = size
        self._get_data()
        
    def _get_data(self):
        data = np.load("./data/ToyData/data.npz", allow_pickle=True)
        self.data_x, self.data_y = data[self.flag]

    def __getitem__(self, index):
        if self.label_len>0:
            y = np.concatenate([self.data_x[index][-self.label_len:], self.data_y[index]], axis=0)
        else:
            y = self.data_y[index]
        return self.data_x[index], self.data_y[index]
    def __len__(self):
        return len(self.data_x)
    def inverse_transform(self, x):
        return x


class ToyDatasetSeq2Seq(DatasetBase):
    def __init__(self,data_path="", flag='train', size=None, 
                 features='S', file_name='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        super().__init__(data_path, size, features, file_name, 
                        target, scale, inverse, timeenc, freq, cols)
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self._get_data()
        
    def _get_data(self):
        data = np.load("./data/ToyData/data.npz", allow_pickle=True)
        self.data_x, self.data_y = data[self.flag]

    def __getitem__(self, index):
        if self.label_len>0:
            y = np.concatenate([self.data_x[index][-self.label_len:], self.data_y[index]], axis=0)
        else:
            y = self.data_y[index]
        return self.data_x[index], y, self.data_x[index], y
    def __len__(self):
        return len(self.data_x)
    def inverse_transform(self, x):
        return x

