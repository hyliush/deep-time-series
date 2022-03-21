from torch import optim
from utils.loss import Normal_loss
import torch.nn as nn
import torch
import os
from utils.loss import OZELoss

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

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
        if isinstance(data, torch.Tensor):
            return data.float().to(self.device)
        if isinstance(data, tuple):
            data = list(data)
        for i in range(len(data)):
            data[i] = data[i].float().to(self.device)
        return data

    def _build_model(self):
        raise NotImplementedError
        return None

    def _process_one_batch(self):
         raise NotImplementedError
         return
    def process_one_batch(self, dataset_object, batch):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, batch_y = self._process_one_batch(batch)
        else:
            outputs, batch_y = self._process_one_batch(batch)
        if self.args.output_hidden:
            outputs = outputs[0]
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
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