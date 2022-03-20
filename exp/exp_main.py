from exp.exp_basic import Exp_Basic
from models.Gdnn import Gdnn
from models.TCN import TCN
from models.TPA import TPA
from models.Trans import Trans
from models.seq2seq import Informer, Autoformer, Transformer, GruAttention, Gru, Lstm
from models.DeepAR import DeepAR
from models.Lstm import BenchmarkLstm
from models.Mlp import BenchmarkMlp
from utils import logger

import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

class Exp_model(Exp_Basic):
    def __init__(self, args):
        Exp_model.init_process_one_batch(args)
        super().__init__(args)

    @classmethod    
    def init_process_one_batch(cls, args):
        if 'former' in args.model:
            cls._process_one_batch = _process_one_batch2
        elif 'ed' in args.model:
            cls._process_one_batch = _process_one_batch5
        elif args.model == "gdnn":
            cls._process_one_batch = _process_one_batch3
        elif args.model == "deepar":
            cls._process_one_batch = _process_one_batch4
        else:
            cls._process_one_batch = _process_one_batch1

    def _build_model(self):
        model_dict = {
            'edlstm': Lstm,
            'edgru': Gru,
            'edgruattention':GruAttention,
            'informer':Informer,
            'transformer': Transformer,
            'autoformer': Autoformer,
            'mlp':BenchmarkMlp,
            'lstm':BenchmarkLstm,
            'tcn':TCN,
            'tpa':TPA,
            'trans':Trans,
            'gated':Gdnn,
            'deepar':DeepAR
        }
        if self.args.model=='gated':
            model = model_dict[self.args.model](
                self.args.n_spatial,
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
            self.args.e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

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
def _process_one_batch4(self, dataset_object, batch):
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


def _process_one_batch5(self, dataset_object, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(self.device)
    batch_y = batch_y.float().to(self.device)

    batch_x_mark = batch_x_mark.float().to(self.device)
    batch_y_mark = batch_y_mark.float().to(self.device)

    # encoder - decoder
    if self.args.use_amp:
        with torch.cuda.amp.autocast():
            if self.args.output_hidden:
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    else:
        if self.args.output_hidden:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
        else:# debug into
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    if self.args.inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if self.args.features=='MS' else 0
    batch_y = batch_y[:,-self.args.pred_len:,f_dim:]

    return outputs, batch_y