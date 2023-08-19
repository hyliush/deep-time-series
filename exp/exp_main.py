from exp.exp_single import Exp_Single
from utils import logger
from torch.utils.data import SubsetRandomSampler
from utils.data import SubsetSequentialSampler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from args import args
from utils.constants import model_dict, dataset_dict
from utils.data import Dataset_Custom
Exp = Exp_Single
import warnings
warnings.filterwarnings('ignore')

class Exp_model(Exp):
    def __init__(self, args, setting, params_dict):
        # for multi
        self.fileName_lst = os.listdir(args.data_path)
        # for single
        self.train_filename, self.val_filename, self.test_filename = \
        [args.file_name]*3 if isinstance(args.file_name, str) else args.file_name

        self.tmp_dataset = None
        self.params_dict = params_dict
        Exp_model.init_process_one_batch(args)
        super().__init__(args, setting)

    @classmethod    
    def init_process_one_batch(cls, args):
        if 'former' in args.model:
            cls._process_one_batch = _process_one_batch2
        elif args.model == "deepar":
            cls._process_one_batch = _process_one_batch3
        else:
            cls._process_one_batch = _process_one_batch1

    def _get_data(self, file_name, flag):
        # 判断是否为第一次加载，不常用，一般都是第一次记载
        if  not hasattr(self.tmp_dataset, "file_name") or self.tmp_dataset.file_name != file_name:
            DataSet = dataset_dict.get(self.args.dataset, Dataset_Custom)
            self.args.file_name = file_name
            self.tmp_dataset = DataSet(self.args)
            logger.debug(flag, len(self.tmp_dataset))
            if flag == "train":
                self.dataset = self.tmp_dataset

        if flag == 'test':
            drop_last = False; sampler = SubsetSequentialSampler; batch_size = self.args.batch_size//2
        else:
            drop_last = True; sampler = SubsetRandomSampler; batch_size = self.args.batch_size
        if hasattr(self.tmp_dataset, flag+"_idxs"):
            idxs = getattr(self.tmp_dataset, flag+"_idxs")
        else:
            raise("flag error")

        data_loader = DataLoader(self.tmp_dataset, batch_size=batch_size, sampler=sampler(idxs),
            num_workers=self.args.num_workers, drop_last=drop_last)
        return data_loader

    def _build_model(self):
        if self.args.model=='informer' or self.args.model=='informerstack':
            self.args.e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

def _process_one_batch1(self, batch):
    '''非seq2seq模型'''
    outputs = self.model(*[batch.get(param) for param in self.input_params])
    return outputs, batch.get(self.target_param)

def _process_one_batch2(self, batch):
    '''seq2seq模型'''
    batch_x, batch_x_mark, batch_y, batch_y_mark = \
        [batch.get(param) for param in self.input_params]
    # decoder input
    if self.args.padding==0: # batch_size * (label_len + pred_len) * out_size pred部分被padding
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
    elif self.args.padding==1:
        dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float()
    # encoder - decoder
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return outputs, batch_y

def _process_one_batch3(self, batch):
    train_batch, idx, labels_batch = batch
    batch_size = train_batch.shape[0]

    train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(self.device)  # not scaled
    labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(self.device)  # not scaled
    idx = idx.unsqueeze(0).to(self.device)

    hidden = self.model.init_hidden(batch_size)
    cell = self.model.init_cell(batch_size)

    mu_sigma_lst = []
    for t in range(self.args.train_window):
        # if z_t is missing, replace it by output mu from the last time step
        zero_index = (train_batch[t, :, 0] == 0) # seq_len * batch_size * 1
        if t > 0 and torch.sum(zero_index) > 0:
            train_batch[t, zero_index, 0] = mu_sigma[0][zero_index]
        mu_sigma, hidden, cell = self.model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
        # record mu, sigma, return batch_mu siga etc
        mu_sigma_lst.append(mu_sigma)
    mu_sigma_batch = torch.stack(mu_sigma_lst, dim=0)
    return mu_sigma_batch.permute(1, 0, 2), labels_batch.unsqueeze(dim=-1).permute(1, 0, 2)