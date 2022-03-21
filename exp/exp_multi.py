import os
import torch
import numpy as np
from utils.metrics import metric
from utils import logger
from utils.tools import EarlyStopping, adjust_learning_rate
from tqdm import tqdm
import time
from torch import optim
from utils.loss import Normal_loss
import torch.nn as nn

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
    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        best_model_path = path+'/'+'checkpoint.pth'

        # 读取上次训练模型
        if self.args.load:
            if "checkpoint.pth" in path:
                print("---------------------load last trained model--------------------------")
                self.model.load_state_dict(torch.load(best_model_path))

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_time = time.time()
            epoch_train_steps_count, epoch_train_steps = 0, 0
            total_train_loss = []
            for file_idx, file_name in enumerate(tqdm(self.fileName_lst), 1):
                train_data, train_loader = self._get_data(file_name = file_name, flag = 'train')
                train_steps = len(train_loader)
                epoch_train_steps_count += train_steps

                train_loss = []
                for i, batch in enumerate(train_loader):
                    epoch_train_steps += 1
                    
                    model_optim.zero_grad()
                    batch_out= self.process_one_batch(train_data, batch)
                    loss = criterion(*batch_out)
                    train_loss.append(loss.item())
                    
                    if (i+1) % 1000==0:
                        print("\epoch: {0}, file_idx: {1}, epoch_train_steps: {2},  | loss: {3:.7f}".format(epoch + 1, file_idx, epoch_train_steps, loss.item()))
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                # file_idx
                train_loss = np.average(train_loss)
                total_train_loss.append(train_loss)
                logger.info("Epoch: {} file_idx: {} train_loss: {}".format(epoch+1, file_idx, train_loss))

            total_vali_loss, vali_metrics = self.vali("val", criterion)
            total_test_loss, test_metrics = self.vali("test", criterion)
            total_train_loss = np.average(total_train_loss)
            # epoch损失记录
            logger.info("Epoch: {}, epoch_train_steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f} cost time: {}".format(
                epoch + 1, epoch_train_steps, total_train_loss, total_vali_loss, total_test_loss, time.time()-epoch_time))
            
            early_stopping(total_vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def _vali(self, val_data, val_loader, criterion):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        loss, preds, trues = [], [], []
        for i, batch in enumerate(val_loader):
            pred, true = self.process_one_batch(val_data, batch)
            pred, true = pred.detach().cpu(), true.detach().cpu()
            preds.append(pred); trues.append(true)
            
            _loss = criterion(pred, true)
            loss.append(_loss)
            
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        loss = np.average(loss)
        return preds, trues, loss

    def vali(self, flag, criterion):
        
        total_loss, total_preds, total_trues = [], [], []
        for file_name in self.fileName_lst:
            val_data, val_loader = self._get_data(file_name, flag=flag)
            preds, trues, loss = self._vali(val_data, val_loader, criterion)
            total_preds.append(preds)
            total_trues.append(trues)
            total_loss.append(loss)

        total_trues, total_preds = np.concatenate(total_trues), np.concatenate(total_preds)
        mae, mse, rmse, mape, mspe = metric(total_preds, total_trues)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss, (mae, mse, rmse, mape, mspe)

    def _test(self, test_data, test_loader, file_path):                
        self.model.eval()
        preds_lst, trues_lst = [], []
        for i, batch in enumerate(test_loader):
            pred, true = self.process_one_batch(test_data, batch)
            preds_lst.append(pred.detach().cpu()); trues_lst.append(true.detach().cpu())
        
        preds, trues = np.concatenate(preds_lst), np.concatenate(trues_lst)
        logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        if file_path is not None:
            np.save(f'{file_path}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(f'{file_path}_pred.npy', preds)
            np.save(f'{file_path}_true.npy', trues)

        return preds, trues

    def test(self, setting, load=False, plot=True):
        # test承接train之后模型，为保证单独使用test，增加load参数
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        total_preds_lst, total_trues_lst = [], []
        
        for file_name in self.fileName_lst:
            test_data, test_loader = self._get_data(file_name, flag='test')

            file_path = folder_path+f'{file_name[:-4]}' if len(self.fileName_lst)>1 else None
            preds, trues = self._test(test_data, test_loader, file_path)
            # inverse
            preds = test_data.inverse_transform(preds)[..., -1:]
            trues = test_data.inverse_transform(trues)[..., -1:]
            total_preds_lst.append(preds)
            total_trues_lst.append(trues)
        
        total_trues, total_preds = np.concatenate(total_trues_lst), np.concatenate(total_preds_lst)
        # total_preds = np.where(abs(total_preds)>10, 0, total_preds)
        # total_trues = np.where(abs(total_trues)>1, 0, total_trues)
        logger.info("test shape:{} {}".format(total_preds.shape, total_trues.shape))
        mae, mse, rmse, mape, mspe = metric(total_preds, total_trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', total_preds)
        np.save(folder_path+f'true.npy', total_trues)
        if plot:
            from utils.visualization import plot_pred, map_plot_function, \
            plot_values_distribution, plot_error_distribution, plot_errors_threshold
            # plot_pred(total_trues, total_preds)
            if self.args.pred_len > 1:
                map_plot_function(total_trues, total_preds, 
                plot_values_distribution, ['volitility'], [0], self.args.pred_len)
            else:
                map_plot_function(total_trues.reshape(120, -1, 1), total_preds.reshape(120, -1, 1), 
                plot_values_distribution, ['volitility'], [0], 6)