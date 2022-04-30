import numpy as np
from soupsieve import Iterable
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import pandas as pd
import torch.nn as nn

def timer(func):
    def deco(*args, **kwargs):
        # print('\n函数：{_funcname_}开始运行：'.format(_funcname_=func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print(end_time - start_time)
        #print('函数:{_funcname_}运行了 {_time_}秒'.format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res
    return deco

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class Writer:
    def __init__(self):
        pass
    def add_scalar(self, *args, **kwargs):
        pass        
    def add_scalars(self, *args, **kwargs):
        pass
    def add_hparams(self, *args, **kwargs):
        pass
    def close(self, *args, **kwargs):
        pass
    def record(self, indictor_name='', y='', x='', type="scalar"):
        # if isinstance(y, dict):
        #     # for key in y.keys():
        #     #     self._record(indictor_name+"_"+key, y[key], x, type)
        # else:
            # self._record(indictor_name, y, x, type)
        # self._record(indictor_name, y, x, type)
        pass

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss # 越大越好
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best_loss:{-self.best_score}, curent_loss:{val_loss}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def filter_extreme(df, type="Percentile", **kwargs):
    '''
    args
    type: Percentile, MAD, Sigma
    kwargs: n_mad, n_sigma, _min, _max
    '''
    flag = 0
    n_mad, n_sigma, _min, _max = kwargs.get("n_mad",5), kwargs.get("n_sigma",3), kwargs.get("_min", 0.1), kwargs.get("_max",0.9)
    train_df = kwargs.get("train_df", df)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        train_df = pd.DataFrame(train_df)
        flag = 1
    if type == "MAD":
        df, min_range, max_range = filter_extreme_MAD(train_df, df, n_mad)
    if type == "Sigma":
        df, min_range, max_range = filter_extreme_3sigma(train_df, df, n_sigma)
    if type == "Percentile":
        df, min_range, max_range = filter_extreme_percentile(train_df, df, _min, _max)
    # if flag == 1:
        # df = df.iloc[:, 0]
    return df, min_range, max_range

def filter_extreme_MAD(train_df, df, n=5): #MAD:中位数去极值
    median = train_df.quantile(0.5)
    new_median = ((train_df - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    df.iloc[:] = np.clip(df,min_range, max_range, axis=1)
    return df, min_range, max_range

def filter_extreme_3sigma(train_df, df, n=3): #3 sigma
    mean = train_df.mean()
    std = train_df.std()
    max_range = mean + n*std
    min_range = mean - n*std
    df.iloc[:] = np.clip(df, min_range, max_range, axis=1)
    return df, min_range, max_range

def filter_extreme_percentile(train_df, df, min = 0.10,max = 0.90): #百分位法
    q = train_df.quantile([min,max])
    min_range, max_range = q.iloc[0],q.iloc[1]
    df.iloc[:] = np.clip(df, min_range, max_range, axis=1)
    return df, min_range, max_range

def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）
    axes：原来的第i维对齐新tensor的第axes[i]维；
    ndim：新tensor的维度。
    """
    assert len(axes) == tensor.dim()
    assert ndim or min(axes) >= 0
    ndim = ndim or max(axes) + 1
    indices = [None] * ndim
    for i in axes:
        indices[i] = slice(None)
    return tensor[indices]

def attention_normalize(a, axis=-1, method='softmax'):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == 'softmax':
        return torch.softmax(a, axis=axis)
    else:
        mask = (a > -torch.tensor(float("inf")) / 10).type(torch.float)
        l = torch.maximum(torch.sum(mask, axis=axis, keepdims=True), 1)
        if method == 'squared_relu':
            return torch.relu(a)**2 / l
        elif method == 'softmax_plus':
            return torch.softmax(a * torch.log(l) / np.log(512), axis=axis)
    return a

class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
         3、hidden_*系列参数仅为有条件输入时(conditional=True)使用，
            用于通过外部条件控制beta和gamma。
    """
    def __init__(
        self,
        key_size,
        scale=True,
        offset=True,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs):
        
        super(ScaleOffset, self).__init__(**kwargs)
        self.key_size = key_size
        self.scale = scale
        self.offset = offset
        self.conditional = conditional
        self.hidden_units = hidden_units

        if self.offset is True:
            self.beta = nn.Parameter(torch.zeros(self.key_size,))

        if self.scale is True:
            self.gamma = self.beta = nn.Parameter(torch.ones(self.key_size,))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Sequential(
                    nn.Linear(self.hidden_units, self.hidden_units, bias=False),
                    hidden_activation)

            if self.offset is not False and self.offset is not None:
                self.beta_dense = nn.Linear(self.key_size, self.key_size, bias=False)
                self.beta_dense.weight = nn.Parameter(torch.zeros(self.key_size, self.size))

            if self.scale is not False and self.scale is not None:
                self.gamma_dense = nn.Linear(self.key_size, self.key_size, bias=False)
                self.gamma_dense.weight = nn.Parameter(torch.zeros(self.key_size, self.size))

    def forward(self, inputs):
        """如果带有条件，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs
            if self.hidden_units is not None:
                conds = self.hidden_dense(conds)
            conds = align(conds, [0, -1], inputs.dim())

        if self.scale is not False and self.scale is not None:
            gamma = self.gamma if self.scale is True else self.scale
            if self.conditional:
                gamma = gamma + self.gamma_dense(conds)
            inputs = inputs * gamma

        if self.offset is not False and self.offset is not None:
            beta = self.beta if self.offset is True else self.offset
            if self.conditional:
                beta = beta + self.beta_dense(conds)
            inputs = inputs + beta

        return inputs