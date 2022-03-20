import os
import torch
import numpy as np
from utils.metrics import metric
from mylogger import logger
from utils.tools import EarlyStopping, adjust_learning_rate
from tqdm import tqdm
import time
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, UbiquantDataSetNoraml, VolatilityDataSetSeq2Seq, VolatilityDataSetNoraml
from torch.utils.data import DataLoader
from torch import optim
from utils.loss import Normal_loss
import torch.nn as nn

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.fileName_lst = os.listdir(args.data_path)
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

    def _move2device(self, *args):
        for i in args:
            i = i.to(self.device)
        return args

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self, file_name, flag):
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
            'Ubiquant':UbiquantDataSetNoraml
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

    def _process_one_batch(self):
        raise NotImplementedError
        return

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
                    batch_out= self._process_one_batch(train_data, batch)
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

    def _vali(self, test_data, test_loader, criterion):
        # 区别于_test, 不需要保存和loss测度
        self.model.eval()
        
        loss, preds, trues = [], [], []
        for i, batch in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch)
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
            test_data, test_loader = self._get_data(file_name, flag=flag)
            preds, trues, loss = self._vali(test_data, test_loader, criterion)
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
            preds, trues = [], []
            for i, batch in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch)
                preds.append(pred.detach().cpu()); trues.append(true.detach().cpu())
            
            preds, trues = np.concatenate(preds), np.concatenate(trues)
            logger.debug('test shape:{} {}'.format(preds.shape, trues.shape))

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

            if file_path is not None:
                np.save(f'{file_path}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(f'{file_path}_pred.npy', preds)
                np.save(f'{file_path}_true.npy', trues)

            return preds, trues

    def test(self, setting, load=False, plot=True):
        # test 比 predict功能更多，但test承接train之后模型，为保证单独使用test，增加load参数
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        total_preds, total_trues = [], []
        
        for file_name in self.fileName_lst:
            test_data, test_loader = self._get_data(file_name, flag='test')

            file_path = folder_path+f'{file_name[:-4]}' if len(self.fileName_lst)>1 else None
            preds, trues = self._test(test_data, test_loader, file_path)
            # inverse
            preds = test_data.inverse_transform(preds)[..., -1:]
            trues = test_data.inverse_transform(trues)[..., -1:]
            total_preds.append(preds)
            total_trues.append(trues)
        
        total_trues, total_preds = np.concatenate(total_trues), np.concatenate(total_preds)
        # total_preds = np.where(abs(total_preds)>10, 0, total_preds)
        # total_trues = np.where(abs(total_trues)>1, 0, total_trues)
        logger.info("test shape:{} {}".format(total_preds.shape, total_trues.shape))
        mae, mse, rmse, mape, mspe = metric(total_preds, total_trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', total_preds)
        np.save(folder_path+f'true.npy', total_trues)
        if plot:
                from utils.visualize import plot_pred
                plot_pred(total_trues, total_preds)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return