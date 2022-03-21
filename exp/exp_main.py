from exp.exp_multi import Exp_Multi
from exp.exp_single import Exp_Single
from models.Gdnn import Gdnn
from models.TCN import TCN
from models.TPA import TPA
from models.Trans import Trans
from models.seq2seq import Informer, Autoformer, Transformer, GruAttention, Gru, Lstm
from models.DeepAR import DeepAR
from models.Lstm import BenchmarkLstm
from models.Mlp import BenchmarkMlp
from utils import logger
from utils.data import order_split
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, ToyDatasetSeq2Seq, UbiquantDataSetNoraml, VolatilityDataSetSeq2Seq, ToyDataset,VolatilityDataSetNoraml
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')
from args import args
Exp = Exp_Single if args.single_file else Exp_Multi

class Exp_model(Exp):
    def __init__(self, args):
        self.fileName_lst = os.listdir(args.data_path)
        self.file_name = self.fileName_lst[0] if len(self.fileName_lst)<=3 else ""
        Exp_model.init_process_one_batch(args)
        super().__init__(args)
        self.__get_data()
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
        pass
    def __get_data(self):
        from data.data_loader import OzeDataset
        DATASET_PATH = r'D:\IDEA\Spatial-temporal\transformer-series\data\dataset1.npz'
        LABELS_PATH = r'D:\IDEA\Spatial-temporal\transformer-series\data\labels.json'
        BATCH_SIZE = 2
        NUM_WORKERS = 0
        dataset = OzeDataset(DATASET_PATH, LABELS_PATH)

        # Split between train, validation and test
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(
            dataset, (5500, 1000, 1000), generator=torch.Generator().manual_seed(42))
        
        self.dataloader_train = DataLoader(self.dataset_train,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=False
                                    )

        self.dataloader_val = DataLoader(self.dataset_val,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS
                                    )

        self.dataloader_test = DataLoader(self.dataset_test,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS
                                    )
    def _get_data(self, file_name, flag):
        if flag == "train":
            return self.dataset_train, self.dataloader_train
        if flag == "val":
            return self.dataset_val, self.dataloader_val
        if flag == "test":
            return self.dataset_test, self.dataloader_test

    def _get_data1(self, file_name, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'Volatility':VolatilityDataSetNoraml,
            'VolatilitySeq2Seq':VolatilityDataSetSeq2Seq,
            'Ubiquant':UbiquantDataSetNoraml,
            'Toy': ToyDataset,
            'ToySeq2Seq': ToyDatasetSeq2Seq
        }
        Data = data_dict[self.args.data+"Seq2Seq"] if "ed" in self.args.model or "former" in self.args.model else data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            data_path=args.data_path,
            file_name=file_name,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        logger.debug(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

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

def _process_one_batch1(self, batch):
    batch_x, batch_y = self._move2device(batch)
    outputs = self.model(batch_x)
    return outputs, batch_y

def _process_one_batch2(self, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = self._move2device(batch)

    # decoder input
    if self.args.padding==0: # batch_size * (label_len + pred_len) * out_size pred部分被padding
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
    elif self.args.padding==1:
        dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
    # encoder - decoder
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return outputs, batch_y

def _process_one_batch3(self, batch):
    batch_x, batch_x_temporal, batch_x_spatial, batch_y = self._move2device(batch)
    outputs = self.model(batch_x, batch_x_temporal, batch_x_spatial)
    return outputs, batch_y

def _process_one_batch5(self, batch):
    batch_x, batch_y = self._move2device(batch)
    outputs = self.model(batch_x, batch_y)
    return outputs, batch_y

def _process_one_batch4(self, batch):
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