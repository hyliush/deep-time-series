import os
import torch
import numpy as np
import torch.nn as nn
from utils.data.dataloader import SubsetSequentialSampler
from utils.constants import dataset_dict
from utils.data import Dataset_Custom
from utils import logger
from datetime import datetime
from utils.tools import dict2string, adjust_learning_rate, save_obj, addkeystring
import time
from tqdm import tqdm
from exp.exp_basic import Exp_Basic
from utils.visualization import plot_pred, map_plot_function, \
plot_values_distribution, plot_error_distribution, \
    plot_errors_threshold, plot_visual_sample
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
            print("---------------------load last trained model--------------------------")
            self.model.load_state_dict(torch.load(best_model_path))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        train_loader = self._get_data(file_name=self.train_filename, flag='train')
        val_loader = self._get_data(file_name=self.val_filename, flag='val')
        
        val_every = len(train_loader)//self.args.val_num if self.args.val_num>0 else np.inf
        for idx_epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()

            running_loss, train_loss = 0, 0
            pbar = tqdm(train_loader, total=len(train_loader), 
                desc=f"[Epoch {idx_epoch+1:3d}/{self.args.train_epochs}]", disable=not self.args.tqdm)
            for idx_batch, batch in enumerate(pbar, 1):
                self.model_optim.zero_grad()
                batch_out= self.process_one_batch(batch)
                loss = self.criterion(*batch_out)
                running_loss += loss.item()
                
                train_loss = running_loss/idx_batch
                pbar.postfix=f"loss={train_loss:.4f}"
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.model_optim.step()

                if idx_batch % val_every==0:
                    vali_loss, vali_metrics_dict = self.vali(val_loader)
                    self.writer.add_scalar("Loss/train", train_loss, idx_epoch*len(train_loader)+idx_batch)
                    self.writer.add_scalar("Loss/val", vali_loss, idx_epoch*len(train_loader)+idx_batch)
                    for key in vali_metrics_dict:
                        self.writer.add_scalar(f"Val_metrics/{key}", vali_metrics_dict[key], idx_epoch*len(train_loader)+idx_batch)
                    
                    logger.info(f"Epoch: {idx_epoch+1} Step:{idx_batch}| Train Loss: {train_loss:.5f} Vali Loss: {vali_loss:.5f} cost time: {(time.time()-epoch_time)/60:.3f} Val_metrics "+dict2string(vali_metrics_dict, self.show_metrics))
            # earlystop 
            self.early_stopping(vali_metrics_dict[self.earlystop_metrics], self.model, self.model_path)
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(self.model_optim, idx_epoch+1, self.args)
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    @torch.no_grad()
    def vali(self, val_loader):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        preds, trues = [], []
        running_loss = 0
        for idx_batch, batch in enumerate(val_loader, 1):
            pred, true = self.process_one_batch(batch)
            running_loss += self.criterion(pred, true)

            pred, true = pred.detach().cpu(), true.detach().cpu()
            preds.append(pred); trues.append(true)

        preds, trues = np.concatenate(preds), np.concatenate(trues)
        loss = running_loss/len(val_loader)
        metrics_dict = self.metrics(preds, trues)

        self.model.train()
        return loss, metrics_dict

    @torch.no_grad()
    def test(self, plot=True, save=False, writer=True, load=True):
        # test承接train之后模型，为保证单独使用test，增加load参数
        test_loader = self._get_data(file_name=self.test_filename, flag='test')
        # DataSet = dataset_dict.get(self.args.dataset, Dataset_Custom)
        # self.tmp_dataset = DataSet(self.args)
        # if self.args.dataset == "mydata":
        #     idxs = self.tmp_dataset.train_idxs.tolist() + self.tmp_dataset.val_idxs.tolist() + self.tmp_dataset.test_idxs.tolist()
        #     test_loader = torch.utils.data.DataLoader(self.tmp_dataset, batch_size=self.args.batch_size,
        #     drop_last=False, sampler=SubsetSequentialSampler(sorted(idxs)))
        # else:
        #     test_loader = torch.utils.data.DataLoader(self.tmp_dataset, batch_size=self.args.batch_size,
        #     drop_last=False)
        if load:
            best_model_path = self.model_path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds_lst, trues_lst = [], []
        for idx_batch, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader), desc="Test"):
            pred, true = self.process_one_batch(batch)
            preds_lst.append(pred.detach().cpu()); trues_lst.append(true.detach().cpu())
        
        preds = np.concatenate(preds_lst)
        trues = np.concatenate(trues_lst)
        logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))
        
        if self.args.out_inverse:
            preds = test_loader.dataset.inverse_transform(preds)[:,:, -1:]
            trues = test_loader.dataset.inverse_transform(trues)[:,:, -1:]
        metrics_dict = self.metrics(preds, trues)
        logger.info(dict2string(metrics_dict, self.show_metrics))

        f = open(f"./results/{self.args.dataset}/result.txt", 'a')
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") +self.setting + "  \n")
        f.write(dict2string(metrics_dict, self.show_metrics))
        f.write('\n')
        f.write('\n')
        f.close()

        if writer:
            self.writer.add_hparams(hparam_dict=self.params_dict, metric_dict=addkeystring(metrics_dict, "Test_metrics/"))
            self.writer.close()

        if save:
            save_obj(os.path.join(self.result_path, f"true_pred.pkl"), [trues, preds])

        if plot:
            from utils.metrics import CORR
            from sklearn.metrics import r2_score
            CORR(trues.flatten(), preds.flatten())
            fig = plot_pred(trues, preds, pred_idx=0, col_idx=-1)
            return fig
            fig.savefig(f"./img/{self.args.model}_{self.args.dataset}_horizon{self.args.horizon}.jpg", bbox_inches='tight')

            if self.args.pred_len > 1:
                idx_labels = range(len(labels))
                labels =[self.args.dataset] * len(idx_labels)
                
                fig = map_plot_function(trues, preds, plot_visual_sample, labels, idx_labels, 168)
                fig.savefig(f"./img/{self.args.model}_{self.args.dataset}_sample.jpg", bbox_inches='tight')

                fig = map_plot_function(trues, preds, plot_values_distribution, labels, idx_labels, 48)
                fig.savefig(f"./img/{self.args.model}_{self.args.dataset}_distribution.jpg", bbox_inches='tight')
            # else:
            #     map_plot_function(trues.reshape(60, -1, 1), preds.reshape(60, -1, 1), 
            #     plot_values_distribution, labels, idx_labels, 6)