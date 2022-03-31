from torch import optim
from utils.loss import Normal_loss
import torch.nn as nn
import torch
import os
from utils.loss import OZELoss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Exp_Basic(object):
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self._init_path(setting)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.writer = SummaryWriter(log_dir = os.path.join(self.run_path,
            '{}'.format(str(datetime.now().strftime('%Y-%m-%d %H-%M-%S')))))
        # self.writer = SummaryWriter(log_dir = self.run_path)

    def _init_path(self, setting):
        self.model_path = os.path.join("./checkpoints/", setting)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        self.result_path = os.path.join("./results/", setting)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.run_path = os.path.join("./runs/", setting)
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.criterion == "mse":
            criterion =  nn.MSELoss()
        if self.args.criterion == "normal":
            criterion = Normal_loss
        if self.args.criterion == "oze":
            criterion = OZELoss(alpha=0.3)
        return criterion

    def _move2device(self, data):
        if isinstance(data, tuple):
            data = list(data)
        if isinstance(data, torch.Tensor):
            return data.float().to(self.device)
        for i in range(len(data)):
            data[i] = self._move2device(data[i])
        return data

    def _build_model(self):
        raise NotImplementedError
        return None

    def _process_one_batch(self):
         raise NotImplementedError
         return
    def process_one_batch(self, batch):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, batch_y = self._process_one_batch(batch)
        else:
            outputs, batch_y = self._process_one_batch(batch)
        if self.args.output_hidden:
            outputs = outputs[0]
        if self.args.inverse:
            outputs = self.dataset.inverse_transform(outputs)
        f_dim = [self.args.target_pos] if self.args.features=='MS' else ...
        batch_y = batch_y[:,-self.args.pred_len:,f_dim]
        outputs = outputs[:,-self.args.pred_len:,f_dim]
        return outputs, batch_y

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device