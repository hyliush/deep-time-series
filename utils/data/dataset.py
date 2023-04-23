import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from fastNLP import cache_results
from utils.timefeatures import time_features
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
import random 
from datetime import datetime, timedelta
from utils.tools import filter_extreme, save_obj, StandardScaler, timer
import json
from typing import Optional

class DatasetBase(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.seq_len, self.label_len, self.pred_len = args.seq_len, args.label_len, args.pred_len
        self.features = args.features
        self.file_name = args.file_name
        self.target = args.target
        self.scale = args.scale
        self.inverse = args.inverse
        self.freq = args.freq
        self.cols = args.cols
        self.date_col = args.date_col
        self.start_col = args.start_col
        self.horizon = args.horizon
        # 未来预测的长度
        self.out_len = max(self.horizon, self.pred_len)
        # seq_len+out_len为一个样本的长度，-1是因为该样本可用
        self.delta = self.seq_len+self.out_len-1
        self.timeenc = 0 if args.embed!='timeF' else 1
        
    def get_idxs1(self, total_len, start_point, shuffle=False):
        length = total_len - self.delta
        test_size = int(length*self.test_size) if isinstance(self.test_size, float) else self.test_size
        val_size = int(length*self.val_size) if isinstance(self.val_size, float) else self.val_size

        cut_point1, cut_point2 = length-test_size-val_size, length-test_size
        border1s = [i+start_point for i in [0, cut_point1, cut_point2]]
        border2s = [i+start_point for i in [cut_point1, cut_point2, length]]
        train_idxs = np.arange(border1s[0], border2s[0]).tolist()
        val_idxs = np.arange(border1s[1], border2s[1]).tolist()
        train_val_idxs = train_idxs + val_idxs
        if shuffle:
            random.seed(125)
            random.shuffle(train_val_idxs)
        train_idxs, val_idxs = train_val_idxs[:len(train_idxs)], train_val_idxs[len(train_idxs):]
        test_idxs = np.arange(border1s[2], border2s[2])
        return train_idxs, val_idxs, test_idxs

    def get_idxs2(self, date, span=None):
        date = pd.to_datetime(date)
        if span:
            # max_span 4
            start_date = f"{self.test_year-2-min(span, 4)+1}-01-01"
        else:
            start_date = "2012-01-01"
        train_date = date[(date>=start_date)&(date<=f"{self.test_year-2}-12-31")][:-self.delta]
        val_date = date[(date>train_date.values[-1]) & (date<=f"{self.test_year-1}-12-31")][:-self.delta]
        test_date = date[(date>val_date.values[-1]) & (date<=f"{self.test_year}-12-31")][:-self.delta]

        train_idxs  = date.index[np.where(date.isin(train_date))[0]];
        val_idxs = date.index[np.where(date.isin(val_date))[0]]
        test_idxs = date.index[np.where(date.isin(test_date))[0]]

        return train_idxs, val_idxs, test_idxs

    def reindex_col(self, df):
        '''
        df.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
        else:
            cols = list(df.columns)
        if self.target:
            # 如果有target，意味着需要重排cols
            cols.remove(self.target)
            cols = cols + [self.target]

        df = df[cols]
        return df
        
    def generate_datefeature(self, df):
        '''如果有date_col，则生成date特征'''
        if self.date_col:
            df = df.rename(columns={self.date_col:"date"})
            df_stamp = df[["date"]]
            df_stamp["date"] = pd.to_datetime(df_stamp["date"])
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        else:
            data_stamp = None
        return data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.horizon-1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        if self.data_stamp is not None:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return dict(zip(["x", "y", "x_mark", "y_mark"],[seq_x, seq_y, seq_x_mark, seq_y_mark]))
        else:
            return dict(zip(["x", "y"],[seq_x, seq_y]))
    
    def __len__(self):
        return len(self.data_x) - self.delta

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df_raw = self.reindex_col(df_raw)
        self.data_stamp = self.generate_datefeature(df_raw)

        border1s = [0, 12*30*24, 12*30*24+4*30*24]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        self.train_idxs = np.arange(border1s[0], border2s[0]-self.delta)
        self.val_idxs = np.arange(border1s[1], border2s[1]-self.delta)
        self.test_idxs = np.arange(border1s[2], border2s[2]-self.delta)

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # scale处理后为data，未处理为df_data
        if self.scale:
            train_data = df_data.iloc[self.train_idxs]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data

class Dataset_ETT_minute(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
        df_raw = self.reindex_col(df_raw)
        self.data_stamp = self.generate_datefeature(df_raw)

        border1s = [0, 12*30*24*4, 12*30*24*4+4*30*24*4]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        self.train_idxs = np.arange(border1s[0], border2s[0]-self.delta)
        self.val_idxs = np.arange(border1s[1], border2s[1]-self.delta)
        self.test_idxs = np.arange(border1s[2], border2s[2]-self.delta)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[self.start_col:]
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

class Dataset_Custom(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.val_size, self.test_size = 0.1, 0.2
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_path,
                                          self.file_name))
        df_raw = self.reindex_col(df_raw)
        self.data_stamp = self.generate_datefeature(df_raw)

        self.train_idxs, self.val_idxs, self.test_idxs = \
            self.get_idxs1(len(df_raw), 0, False)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data.iloc[self.train_idxs]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data

class Dataset_Pred(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
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
            cols_data = df_raw.columns[self.start_col:]
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


class MyDataSet(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.test_year = self.args.test_year
        self.__read_data__()

    def __read_data__(self):
        @timer
        @cache_results(_cache_fp=None)
        def _get_data(shuffle=False, span=None):
            # @cache_results(os.path.join(self.data_path, self.file_name[:-4], '.pkl'))
            df_raw = pd.read_csv(os.path.join(self.data_path, self.file_name))
            date_lst = [df_raw["Date"][df_raw["stock_id"]==i] 
                        for i in df_raw["stock_id"].unique()]
            _tmp = Parallel(n_jobs=-1)(delayed(self.get_idxs2)(date, span) for date in date_lst)

            train_idxs, val_idxs, test_idxs = zip(*_tmp)
            train_idxs, val_idxs, test_idxs = np.concatenate(train_idxs), np.concatenate(val_idxs), np.concatenate(test_idxs)
            
            self.scaler = StandardScaler()
            df_raw = self.reindex_col(df_raw)
            df_raw = df_raw.rename(columns={"Date":"date"})
            df_stamp = self.generate_datefeature(df_raw)

            if isinstance(self.features, str):
                if self.features=='M' or self.features=='MS':
                    cols_data = df_raw.columns[self.start_col:]
                    df_data = df_raw[cols_data]
                elif self.features=='S':
                    df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data.iloc[train_idxs]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            if self.inverse:
                data_y = df_data.values
            else:
                data_y = data
            return self.scaler, df_stamp, data, data_y, train_idxs, val_idxs, test_idxs

        shuffle = False
        self.scaler, self.data_stamp, self.data_x, self.data_y, \
        self.train_idxs, self.val_idxs, self.test_idxs = \
        _get_data(shuffle=shuffle, span=4, _cache_fp=os.path.join('./cache', 
        f"{self.file_name[:-4]}_{self.args.dataset}_sl{self.seq_len}_pl{self.pred_len}_ty{self.test_year}_hn{self.horizon}_sf{int(shuffle)}_{self.args.des}_span4.pkl"))


class MyDataSetGate(Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.test_size = 60
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
                cols_data = df_raw.columns[self.start_col:]
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

        return {"x":self._x[idx], "y":self._y[idx]}
    def __len__(self):
        return self._x.shape[0]


class ToyDataset(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.test_size = 60
        self.__read_data__()
        
    def __read_data__(self):
        data = np.load("./data/ToyData/data.npz", allow_pickle=True)
        self.data_x, self.data_y = data[self.flag]

    def __getitem__(self, index):
        if self.label_len>0:
            y = np.concatenate([self.data_x[index][-self.label_len:], self.data_y[index]], axis=0)
        else:
            y = self.data_y[index]
        return {"x":self.data_x[index], "y":self.data_y[index]}
    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, x):
        return x

class MyDataset1(DatasetBase):
    def __init__(self, args):
        super().__init__(args)
        self.val_size, self.test_size = 0.1, 0.2
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(f"./data/{self.args.file_name}.csv", index_col=0)
        df = df[["stock_id"]+[self.target]]

        sample_id_lst = [np.arange(len(df))[df["stock_id"]==i] 
                    for i in df["stock_id"].unique()]
        _tmp = Parallel(n_jobs=-1)(delayed(self.get_idxs1)(len(x), x[0], False) for x in sample_id_lst)

        train_idxs, val_idxs, test_idxs = zip(*_tmp)
        self.train_idxs, self.val_idxs, self.test_idxs = np.concatenate(train_idxs), np.concatenate(val_idxs), np.concatenate(test_idxs)
        save_obj(f"./save_file/idxs{self.out_len}.pkl", [self.train_idxs, self.val_idxs, self.test_idxs])
        df = df.drop(columns=["stock_id"])
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df#.iloc[self.train_idxs]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df.values)
        else:
            data = df.values
        self.data_x = data

        if self.inverse:
            self.data_y = df.values
        else:
            self.data_y = data
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.horizon-1
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        return dict(zip(["x", "y"],[seq_x, seq_y]))


def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
    # for i in range(1, len(data)):
        row = data[i].replace('"', '').split(',')
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids.append(row[0])

    series = np.array(series)
    return series


def create_val_set(train, output_size):
    val = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        train[i] = train[i][:-output_size]
    return np.array(val)


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


def create_datasets(train_file_location, test_file_location, output_size):
    train = read_file(train_file_location)
    test = read_file(test_file_location)
    val = create_val_set(train, output_size)
    return train, val, test


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, info, variable, chop_value, device):
        dataTrain, mask = chop_series(dataTrain, chop_value)

        self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category'])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal)) if mask[i]]
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest)) if mask[i]]
        self.device = device
        self.dataTrain = torch.stack(self.dataTrain, axis=1)
        self.dataVal = torch.stack(self.dataVal, axis=1)
        self.dataTest = torch.stack(self.dataTest, axis=1)

        self.scaler = StandardScaler()
        self.scaler.transform()
    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
               self.dataVal[idx].to(self.device), \
               self.dataTest[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device), \
               idx

    def inverse_transform(self, data):
        pass        
    def inverse(self, data):
        pass


