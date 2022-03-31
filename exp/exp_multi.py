import os
import torch
import numpy as np
from utils.metrics import metric
from utils import logger
from utils.tools import EarlyStopping, adjust_learning_rate
from tqdm import tqdm
import time
from exp.exp_basic import Exp_Basic
from utils.visualization import plot_pred, map_plot_function, \
plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample
class Exp_Multi(Exp_Basic):
    def __init__(self, args, setting):
        super().__init__(args, setting)

    def train(self):
        best_model_path = self.model_path+'/'+'checkpoint.pth'

        # 读取上次训练模型
        if self.args.load:
            if "checkpoint.pth" in self.model_path:
                print("---------------------load last trained model--------------------------")
                self.model.load_state_dict(torch.load(best_model_path))

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for idx_epoch in range(self.args.train_epochs):
            self.model.train()

            epoch_time = time.time()
            epoch_train_steps_count, epoch_train_steps = 0, 0
            total_train_loss = []
            for file_idx, file_name in enumerate(tqdm(self.fileName_lst), 1):
                train_loader = self._get_data(file_name = file_name, flag = 'train')
                print_every = len(train_loader)//self.args.print_num
                train_steps = len(train_loader)
                epoch_train_steps_count += train_steps

                running_loss = 0
                for idx_batch, batch in enumerate(train_loader):
                    epoch_train_steps += 1
                    
                    model_optim.zero_grad()
                    batch_out= self.process_one_batch(batch)
                    loss = criterion(*batch_out)
                    running_loss += loss.item()
                    
                    if (idx_batch+1) % print_every==0:
                        logger.info("Epoch: {0}, file_idx: {1}, epoch_train_steps: {2},  | loss: {3:.7f}".format(idx_epoch + 1, file_idx, epoch_train_steps, loss.item()))
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                # file_idx
                train_loss = running_loss/len(train_loader)
                total_train_loss.append(train_loss)
                logger.info("Epoch: {} file_idx: {} train_loss: {}".format(idx_epoch+1, file_idx, train_loss))

            total_vali_loss, vali_metrics_dict = self.vali("val", criterion)
            total_train_loss = np.average(total_train_loss)
            # epoch损失记录
            logger.info("Epoch: {}, epoch_train_steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} cost time: {}".format(
                idx_epoch + 1, epoch_train_steps, total_train_loss, total_vali_loss, (time.time()-epoch_time)/60))
            
            early_stopping(total_vali_loss, self.model, self.model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, idx_epoch+1, self.args)
            
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _vali(self, val_loader, criterion):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        preds, trues = [], []
        running_loss = 0
        for i, batch in enumerate(val_loader):
            pred, true = self.process_one_batch(batch)
            pred, true = pred.detach().cpu(), true.detach().cpu()
            preds.append(pred); trues.append(true)
            
            _loss = criterion(pred, true)
            running_loss += _loss.item()
            
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        loss = running_loss/len(val_loader)
        return preds, trues, loss

    def vali(self, flag, criterion):
        
        total_loss, total_preds, total_trues =[], [], []
        for file_name in self.fileName_lst:
            val_loader = self._get_data(file_name, flag=flag)
            preds, trues, loss = self._vali(val_loader, criterion)
            total_preds.append(preds)
            total_trues.append(trues)
            total_loss.append(loss)

        total_trues, total_preds = np.concatenate(total_trues), np.concatenate(total_preds)
        metrics_dict = metric(total_preds, total_trues)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss, metrics_dict

    def _test(self, test_loader, file_path):                
        self.model.eval()
        preds_lst, trues_lst = [], []
        for i, batch in enumerate(test_loader):
            pred, true = self.process_one_batch(batch)
            preds_lst.append(pred.detach().cpu()); trues_lst.append(true.detach().cpu())
        
        preds, trues = np.concatenate(preds_lst), np.concatenate(trues_lst)
        logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))
        
        metrics_dict = metric(preds, trues)
        logger.info('mse:{}, mae:{}'.format(metrics_dict["mse"], metrics_dict["mae"]))

        if file_path is not None:
            np.save(f'{file_path}_pred.npy', preds)
            np.save(f'{file_path}_true.npy', trues)

        return preds, trues

    def test(self, load=False, plot=True, save=False):
        # test承接train之后模型，为保证单独使用test，增加load参数
        if load:
            best_model_path = self.model_path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        total_preds_lst, total_trues_lst = [], []
        
        for file_name in self.fileName_lst:
            test_loader = self._get_data(file_name, flag='test')

            file_path = self.result_path+f'{file_name[:-4]}'
            preds, trues = self._test(test_loader, file_path)
            # inverse
            if self.args.inverse:
                preds = test_loader.dataset.inverse_transform(preds)[..., -1:]
                trues = test_loader.dataset.inverse_transform(trues)[..., -1:]
            total_preds_lst.append(preds)
            total_trues_lst.append(trues)
        
        total_trues, total_preds = np.concatenate(total_trues_lst), np.concatenate(total_preds_lst)
        # total_preds = np.where(abs(total_preds)>10, 0, total_preds)
        # total_trues = np.where(abs(total_trues)>1, 0, total_trues)
        logger.info("test shape:{} {}".format(total_preds.shape, total_trues.shape))
        metrics_dict = metric(total_preds, total_trues)
        logger.info('mse:{}, mae:{}'.format(metrics_dict["mse"], metrics_dict["mae"]))
        
        if save:
            np.save(self.result_path+'pred.npy', preds)
            np.save(self.result_path+f'true.npy', trues)
        if plot:
            # plot_pred(total_trues, total_preds)
            if self.args.pred_len > 1:
                map_plot_function(total_trues, total_preds, 
                plot_values_distribution, ['volitility'], self.args.pred_len)
            else:
                map_plot_function(total_trues.reshape(120, -1, 1), total_preds.reshape(120, -1, 1), 
                plot_values_distribution, ['volitility'], 6)