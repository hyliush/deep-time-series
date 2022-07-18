from torch import optim
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.tools import Writer, EarlyStopping
import inspect
from utils.constants import criterion_dict
from utils.metrics import point_metric, distribution_metric

class Exp_Basic(object):
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.prefix = "predict" if self.args.do_predict else "train"
        self._init_path(setting)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion =  criterion_dict[self.args.criterion]
        self.model_optim = self._select_optimizer()
        self.input_params = args.input_params or inspect.signature(self.model.forward).parameters.keys()
        self.target_param = args.target_param
        self.writer = SummaryWriter(log_dir = os.path.join(self.run_path,
            f"{self.prefix}_{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")) if not args.debug else Writer()
        # self.writer = SummaryWriter(log_dir = self.run_path)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if self.args.criterion ==  "mse":
            self.metric = point_metric
            self.show_metrics = ["mse", "r2"]
        else:
            self.metric = distribution_metric
            self.show_metrics = ["rho50", "rho90"]

    def _init_path(self, setting):
        self.model_path = os.path.join(f"./checkpoints/{self.args.dataset}/", setting)
        self.result_path = os.path.join(f"./results/{self.args.dataset}/", setting)
        self.run_path = os.path.join(f"./runs/{self.args.dataset}/{self.prefix}/", setting)
        if not self.args.debug:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)

            if not os.path.exists(self.run_path):
                os.makedirs(self.run_path)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _move2device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.float().to(self.device)
        if isinstance(obj, tuple):
            obj = list(obj)
            obj = self._move2device(obj)
            return obj
        if isinstance(obj, dict):
            for key in obj.keys():
                obj[key] = self._move2device(obj[key])
            return obj
        if isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._move2device(obj[i])
            return obj

    def _build_model(self):
        raise NotImplementedError
        return None

    def _process_one_batch(self):
         raise NotImplementedError
         return
    def process_one_batch(self, batch):
        batch = self._move2device(batch)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, batch_y = self._process_one_batch(batch)
        else:
            outputs, batch_y = self._process_one_batch(batch)
        if self.args.output_attention:
            outputs = outputs[0]
        if self.args.inverse:
            outputs = self.dataset.inverse_transform(outputs)
        f_dim = [-1] if self.args.features=='MS' else ...
        batch_y = batch_y[:,-self.args.pred_len:,f_dim]
        # probabilistic 
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