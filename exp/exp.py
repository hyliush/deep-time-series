from exp.exp_basic import Exp_Basic
from models.gdnn import gdnn
from models.TCN import TCN
from models.TPA import TPA
from models.Trans import Trans
from models.informer import Informer, InformerStack
from models.lstm import BenchmarkLSTM
from models.mlp import BenchmarkMlp
from mylogger import logger

import torch
import torch.nn as nn
from torch import optim
import os
import warnings
warnings.filterwarnings('ignore')

class Exp_model(Exp_Basic):
    def __init__(self, args):
        self.fileName_lst = os.listdir(args.data_path)
        Exp_model.init_process_one_batch(args)
        super().__init__(args)

    @classmethod    
    def init_process_one_batch(cls, args):
        if args.model == "informer" or args.model == "informerstack":
            cls._process_one_batch = _process_one_batch2
        elif args.model == "gdnn":
            cls._process_one_batch = _process_one_batch3
        else:
            cls._process_one_batch = _process_one_batch1

    def _build_model(self):
        model_dict = {
            'tcn':TCN,
            'mlp':BenchmarkMlp,
            'tpa':TPA,
            'lstm':BenchmarkLSTM,
            'trans':Trans,
            'informer':Informer,
            'informerstack':InformerStack,
            'gated':gdnn
        }
        if self.args.model=='gated':
            model = model_dict[self.args.model](
                self.args.c_in,
                self.args.gdnn_embed_size,
                self.args.embed,
                self.args.freq,
                self.args.input_size,
                self.args.gdnn_hidden_size1,
                self.args.gdnn_out_size,
                self.args.gdnn_n_layers,
                self.args.gdnn_hidden_size2,
                self.args.out_size,
            ).float()

        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.out_size, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_hidden,
                self.args.distil,
                self.args.mix
            ).float()

        if self.args.model=='tcn':
            model = model_dict[self.args.model](
                self.args.input_size,
                self.args.tcn_hidden_size, 
                self.args.tcn_n_layers,
                self.args.tcn_dropout,
                self.args.out_size
            ).float()
        if self.args.model=='tpa':
            model = model_dict[self.args.model](
                self.args.input_size,
                self.args.seq_len,
                self.args.tpa_hidden_size, 
                self.args.tpa_n_layers,
                self.args.tpa_ar_len,
                self.args.out_size
            ).float()
        if self.args.model=='mlp':
            model = model_dict[self.args.model](
                self.args.input_size,
                self.args.mlp_hidden_size, 
                self.args.out_size
            ).float()

        if self.args.model=='lstm':
            model = model_dict[self.args.model](
                self.args.input_size,
                self.args.lstm_hidden_size, 
                self.args.out_size,
                self.args.lstm_n_layers,
            ).float()
        if self.args.model=='trans':
            model = model_dict[self.args.model](
                self.args.input_size,
                self.args.trans_hidden_size, 
                self.args.trans_kernel_size,
                self.args.pred_len,
                self.args.trans_n_heads,
                self.args.trans_n_layers,
                self.args.out_size
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

def _process_one_batch1(self, dataset_object, batch):
    batch_x = batch[0].float().to(self.device)
    batch_y = batch[1].float()

    if self.args.use_amp:
        with torch.cuda.amp.autocast():
            if self.args.output_hidden:
                outputs = self.model(batch_x)[0]
            else:
                outputs = self.model(batch_x)
    else:
        if self.args.output_hidden:
            outputs = self.model(batch_x)[0]
        else:# debug into
            outputs = self.model(batch_x)
    if self.args.inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if self.args.features=='MS' else 0
    batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

    return outputs, batch_y

def _process_one_batch2(self, dataset_object, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(self.device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(self.device)
    batch_y_mark = batch_y_mark.float().to(self.device)

    # decoder input
    if self.args.padding==0: # batch_size * (label_len + pred_len) * out_size pred部分被padding
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
    elif self.args.padding==1:
        dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
    # encoder - decoder
    if self.args.use_amp:
        with torch.cuda.amp.autocast():
            if self.args.output_hidden:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    else:
        if self.args.output_hidden:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:# debug into
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    if self.args.inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if self.args.features=='MS' else 0
    batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

    return outputs, batch_y

def _process_one_batch3(self, dataset_object, batch):
    batch_x, batch_x_temporal, batch_x_spatial, batch_y = batch
    batch_x = batch_x.float().to(self.device)
    batch_x_temporal = batch_x_temporal.to(self.device)
    batch_x_spatial = batch_x_spatial.to(self.device)
    batch_y = batch_y.float()

    # encoder - decoder
    if self.args.use_amp:
        with torch.cuda.amp.autocast():
            if self.args.output_hidden:
                outputs = self.model(batch_x, batch_x_temporal, batch_x_spatial)[0]
            else:
                outputs = self.model(batch_x, batch_x_temporal, batch_x_spatial)
    else:
        if self.args.output_hidden:
            outputs = self.model(batch_x, batch_x_temporal, batch_x_spatial)[0]
        else:# debug into
            outputs = self.model(batch_x, batch_x_temporal, batch_x_spatial)
    if self.args.inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if self.args.features=='MS' else 0
    batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

    return outputs, batch_y
