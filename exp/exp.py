from exp.exp_basic import Exp_Basic
from models.Gdnn import Gdnn
from models.TCN import TCN
from models.TPA import TPA
from models.Trans import Trans
from models.informer import Informer, InformerStack
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
        self.fileName_lst = os.listdir(args.data_path)
        # for i in ['id1027.csv', 'id1030.csv', 'id1033.csv', 'id1052.csv', 'id1057.csv', 'id1070.csv', 'id1080.csv', 'id1088.csv', 'id1091.csv', 'id1116.csv', 'id1120.csv', 'id1132.csv', 'id1135.csv', 'id1149.csv', 'id1183.csv', 'id1189.csv', 'id1192.csv', 'id1198.csv', 'id1207.csv', 'id1213.csv', 'id1274.csv', 'id1289.csv', 'id1307.csv', 'id1315.csv', 'id1323.csv', 'id1333.csv', 'id1358.csv', 'id1364.csv', 'id1382.csv', 'id1413.csv', 'id1415.csv', 'id1433.csv', 'id1436.csv', 'id1437.csv', 'id1439.csv', 'id1440.csv', 'id1449.csv', 'id1453.csv', 'id1458.csv', 'id1484.csv', 'id1487.csv', 'id1494.csv', 'id15.csv', 'id1505.csv', 'id1518.csv', 'id154.csv', 'id1559.csv', 'id1566.csv', 'id1571.csv', 'id1572.csv', 'id1573.csv', 'id1578.csv', 'id1586.csv', 'id1613.csv', 'id1636.csv', 'id1638.csv', 'id1649.csv', 'id165.csv', 'id168.csv', 'id1680.csv', 'id1706.csv', 'id1732.csv', 'id1759.csv', 'id1815.csv', 'id1831.csv', 'id1841.csv', 'id1843.csv', 'id1859.csv', 'id1866.csv', 'id1871.csv', 'id1873.csv', 'id1881.csv', 'id1896.csv', 'id190.csv', 'id1916.csv', 'id1918.csv', 'id1979.csv', 'id198.csv', 'id1989.csv', 'id2007.csv', 'id201.csv', 'id2017.csv', 'id2041.csv', 'id2067.csv', 'id2101.csv', 'id2105.csv', 'id2123.csv', 'id2135.csv', 'id2139.csv', 'id2190.csv', 'id2194.csv', 'id22.csv', 'id2215.csv', 'id2222.csv', 'id2232.csv', 'id2246.csv', 'id2250.csv', 'id227.csv', 'id2297.csv', 'id2326.csv', 'id2381.csv', 'id2384.csv', 'id239.csv', 'id240.csv', 'id2415.csv', 'id2424.csv', 'id2426.csv', 'id2433.csv', 'id2435.csv', 'id2437.csv', 'id2444.csv', 'id2450.csv', 'id2469.csv', 'id2482.csv', 'id2486.csv', 'id2505.csv', 'id2535.csv', 'id2539.csv', 'id2558.csv', 'id2567.csv', 'id2583.csv', 'id2598.csv', 'id2606.csv', 'id2615.csv', 'id2616.csv', 'id2643.csv', 'id2657.csv', 'id269.csv', 'id2691.csv', 'id2698.csv', 'id2707.csv', 'id2716.csv', 'id2717.csv', 'id272.csv', 'id2721.csv', 'id2730.csv', 'id2745.csv', 'id2746.csv', 'id2764.csv', 'id2766.csv', 'id2767.csv', 'id2787.csv', 'id2800.csv', 'id2811.csv', 'id2826.csv', 'id2889.csv', 'id2917.csv', 'id2918.csv', 'id2974.csv', 'id2981.csv', 'id3003.csv', 'id3005.csv', 'id3058.csv', 'id3116.csv', 'id3129.csv', 'id3141.csv', 'id3153.csv', 'id3169.csv', 'id319.csv', 'id3203.csv', 'id3208.csv', 'id3229.csv', 'id3243.csv', 'id3248.csv', 'id325.csv', 'id3261.csv', 'id3274.csv', 'id3281.csv', 'id330.csv', 'id3339.csv', 'id3349.csv', 'id3363.csv', 'id339.csv', 'id3400.csv', 'id3408.csv', 'id3410.csv', 'id342.csv', 'id3425.csv', 'id3432.csv', 'id3449.csv', 'id3454.csv', 'id3468.csv', 'id3477.csv', 'id349.csv', 'id3557.csv', 'id3570.csv', 'id3588.csv', 'id3598.csv', 'id3602.csv', 'id3603.csv', 'id3611.csv', 'id3612.csv', 'id3617.csv', 'id3624.csv', 'id3630.csv', 'id3642.csv', 'id3648.csv', 'id3649.csv', 'id3662.csv', 'id3669.csv', 'id367.csv', 'id3675.csv', 'id3684.csv', 'id3696.csv', 'id3712.csv', 'id3713.csv', 'id374.csv', 'id3755.csv', 'id3763.csv', 'id427.csv', 'id435.csv', 'id458.csv', 'id472.csv', 'id483.csv', 'id484.csv', 'id487.csv', 'id500.csv', 'id549.csv', 'id550.csv', 'id551.csv', 'id554.csv', 'id559.csv', 'id56.csv', 'id567.csv', 'id57.csv', 'id590.csv', 'id592.csv', 'id60.csv', 'id612.csv', 'id623.csv', 'id624.csv', 'id632.csv', 'id643.csv', 'id649.csv', 'id666.csv', 'id68.csv', 'id709.csv', 'id714.csv', 'id719.csv', 'id72.csv', 'id732.csv', 'id767.csv', 'id776.csv', 'id791.csv', 'id803.csv', 'id817.csv', 'id829.csv', 'id85.csv', 'id854.csv', 'id879.csv', 'id890.csv', 'id905.csv', 'id91.csv', 'id915.csv', 'id931.csv', 'id956.csv', 'id957.csv', 'id980.csv', 'id993.csv']:
        #     self.fileName_lst.remove(i)

        Exp_model.init_process_one_batch(args)
        super().__init__(args)

    @classmethod    
    def init_process_one_batch(cls, args):
        if args.model == "informer" or args.model == "informerstack":
            cls._process_one_batch = _process_one_batch2
        elif args.model == "gdnn":
            cls._process_one_batch = _process_one_batch3
        elif args.model == "deepar":
            cls._process_one_batch = _process_one_batch4
        else:
            cls._process_one_batch = _process_one_batch1

    def _build_model(self):
        model_dict = {
            'tcn':TCN,
            'mlp':BenchmarkMlp,
            'tpa':TPA,
            'lstm':BenchmarkLstm,
            'trans':Trans,
            'informer':Informer,
            'informerstack':InformerStack,
            'gated':Gdnn,
            'deepar':DeepAR
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
