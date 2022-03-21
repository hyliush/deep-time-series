import os
import torch
import numpy as np
from utils.metrics import metric
from utils import logger
from utils.tools import EarlyStopping, adjust_learning_rate
import time
from tqdm import tqdm
from exp.exp_basic import Exp_Basic

class Exp_Single(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

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
        
        train_data, train_loader = self._get_data(file_name=self.file_name, flag='train')
        val_data, val_loader = self._get_data(file_name=self.file_name, flag='val')
        
        for idx_epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()

            running_loss = 0
            with tqdm(total=len(train_loader), desc=f"[Epoch {idx_epoch+1:3d}/{self.args.train_epochs}]") as pbar:
                for idx_batch, batch in enumerate(train_loader):
                    model_optim.zero_grad()
                    batch_out= self.process_one_batch(train_data, batch)
                    loss = criterion(*batch_out)
                    running_loss += loss.item()

                    pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                    pbar.update()

                    if (idx_batch+1) % 100==0:
                        logger.info("Epoch: {0}, epoch_train_steps: {1},  | loss: {2:.7f}".format(idx_epoch+1, idx_batch+1, loss.item()))
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                vali_loss, vali_metrics = self.vali(val_data, val_loader, criterion)
                train_loss = running_loss/len(train_loader)
                # epoch损失记录
                logger.info("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f} cost time: {}".format(
                    idx_epoch + 1, train_loss, vali_loss, time.time()-epoch_time))
                
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, idx_epoch+1, self.args)
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def vali(self, val_data, val_loader, criterion):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        preds, trues = [], []
        running_loss = 0
        for i, batch in enumerate(val_loader):
            pred, true = self.process_one_batch(val_data, batch)
            pred, true = pred.detach().cpu(), true.detach().cpu()
            preds.append(pred); trues.append(true)
            
            running_loss += criterion(pred, true)
            
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        loss = running_loss/len(val_loader)
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        self.model.train()
        return loss, (mae, mse, rmse, mape, mspe)

    def test(self, setting, load=False, plot=True):
        # test承接train之后模型，为保证单独使用test，增加load参数
        test_data, test_loader = self._get_data(file_name=self.file_name, flag='test')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        preds_lst, trues_lst = [], []
        for i, batch in enumerate(test_loader):
            pred, true = self.process_one_batch(test_data, batch)
            preds_lst.append(pred.detach().cpu()); trues_lst.append(true.detach().cpu())
        
        preds, trues = np.concatenate(preds_lst), np.concatenate(trues_lst)
        logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info('mse:{}, mae:{}'.format(mse, mae))

        # inverse
        preds = test_data.inverse_transform(preds)[..., -1:]
        trues = test_data.inverse_transform(trues)[..., -1:]

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+f'true.npy', trues)
        if plot:
            from utils.visualization import plot_pred, map_plot_function, \
            plot_values_distribution, plot_error_distribution, plot_errors_threshold
            # plot_pred(total_trues, preds)
            if self.args.pred_len > 1:
                map_plot_function(trues, preds, 
                plot_values_distribution, ['volitility'], [0], self.args.pred_len)
            else:
                map_plot_function(trues.reshape(120, -1, 1), preds.reshape(120, -1, 1), 
                plot_values_distribution, ['volitility'], [0], 6)