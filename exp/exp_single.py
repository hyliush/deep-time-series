import os
import torch
import numpy as np
from utils.metrics import metric
from utils import logger
from utils.tools import EarlyStopping, adjust_learning_rate
import time
from tqdm import tqdm
from exp.exp_basic import Exp_Basic
from utils.visualization import plot_pred, map_plot_function, \
plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Exp_Single(Exp_Basic):
    def __init__(self, args, setting):
        super().__init__(args, setting)

    def train(self):
        best_model_path = self.model_path+'/'+'checkpoint.pth'
        # 读取上次训练模型
        if self.args.load:
            if "checkpoint.pth" in self.model_path:
                logger.info("---------------------load last trained model--------------------------")
                self.model.load_state_dict(torch.load(best_model_path))

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        train_loader = self._get_data(file_name=self.train_filename, flag='train')
        val_loader = self._get_data(file_name=self.val_filename, flag='val')
        
        for idx_epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()

            running_loss = 0
            with tqdm(total=len(train_loader), desc=f"[Epoch {idx_epoch+1:3d}/{self.args.train_epochs}]") as pbar:
                for idx_batch, batch in enumerate(train_loader, 1):
                    model_optim.zero_grad()
                    batch_out= self.process_one_batch(batch)
                    loss = criterion(*batch_out)
                    running_loss += loss.item()

                    pbar.set_postfix({'loss': running_loss/idx_batch})
                    pbar.update()

                    if idx_batch % self.args.print_every==0:
                        logger.info("Epoch: {0}, epoch_train_steps: {1},  | loss: {2:.7f}".format(idx_epoch+1, idx_batch+1, loss.item()))
                        self.writer.record("Loss/train", running_loss/idx_batch, idx_epoch*len(train_loader)+idx_batch)

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                train_loss = running_loss/len(train_loader)
                vali_loss, vali_metrics = self.vali(val_loader, criterion)
                self.writer.record("Loss/val", vali_loss, idx_epoch+1)
                self.writer.record("Metrics/val", vali_metrics, idx_epoch+1)

                # epoch损失记录
                logger.info("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f} cost time: {}".format(
                    idx_epoch + 1, train_loss, vali_loss, (time.time()-epoch_time)/60))
                
                early_stopping(vali_loss, self.model, self.model_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, idx_epoch+1, self.args)
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def vali(self, val_loader, criterion):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        preds, trues = [], []
        running_loss = 0
        for idx_batch, batch in tqdm(enumerate(val_loader, 1), total=len(val_loader), desc="Validation"):
            pred, true = self.process_one_batch(batch)
            pred, true = pred.detach().cpu(), true.detach().cpu()
            preds.append(pred); trues.append(true)
            
            running_loss += criterion(pred, true)

        preds, trues = np.concatenate(preds), np.concatenate(trues)
        loss = running_loss/len(val_loader)
        metrics = metric(preds, trues)

        self.model.train()
        return loss, metrics

    def test(self, load=False, plot=True, save=False):
        # test承接train之后模型，为保证单独使用test，增加load参数
        test_loader = self._get_data(file_name=self.test_filename, flag='test')
        if load:
            best_model_path = self.model_path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds_lst, trues_lst = [], []
        for idx_batch, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader), desc="Test"):
            pred, true = self.process_one_batch(batch)
            preds_lst.append(pred.detach().cpu()); trues_lst.append(true.detach().cpu())
        
        preds, trues = np.concatenate(preds_lst), np.concatenate(trues_lst)
        logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))
        
        if self.args.inverse:
            preds = np.sqrt(np.exp(test_loader.dataset.inverse_transform(preds)[..., -1:]))
            trues = np.sqrt(np.exp(test_loader.dataset.inverse_transform(trues)[..., -1:]))
        metrics = metric(preds, trues)
        logger.info('mse:{}, mae:{}'.format(metrics["mse"], metrics["mae"]))
        
        if save:
            np.save(self.result_path+'pred.npy', preds)
            np.save(self.result_path+f'true.npy', trues)
        if plot:
            from utils.metrics import CORR
            plot_pred(trues, preds, pred_idx=0, col_idx=-1)
            if self.args.pred_len > 1:
                # labels = test_data.dataset.labels["X"]
                # labels = "HUFL,HULL,MUFL,MULL,LUFL,LULL,OT".split(',')
                idx_labels = [-1]
                labels =["Volatility"] * len(idx_labels)
                fig = map_plot_function(trues, preds, plot_visual_sample, labels, idx_labels, 168)
                # fig.savefig(f"./img/{self.args.model}_sample.jpg", bbox_inches='tight')

                fig = map_plot_function(trues, preds, plot_values_distribution, labels, idx_labels, 48)
                # fig.savefig(f"./img/{self.args.model}_distribution.jpg", bbox_inches='tight')
            else:
                map_plot_function(trues.reshape(60, -1, 1), preds.reshape(60, -1, 1), 
                plot_values_distribution, ['volitility'], 6)