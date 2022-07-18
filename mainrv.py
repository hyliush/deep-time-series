import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import product_args
from utils.tools import get_params_dict
from collections import OrderedDict
import numpy as np
args.dataset = "mydata"
args.file_name = "rv" #"5f5533cpu", "53ea38cpu","mean CPU usage rate", "assigned memory usage"
args.patience = 2 # earlystop
args.do_predict = True 
args.out_inverse = True
# args.scale = False
# LSTM
args.model = "lstmdecompose"
args.seq_len, args.label_len, args.pred_len = 40, 0, 1
args.input_params = ["x"] #指定模型的输入
args.learning_rate = 0.001
args.lradj = "type10" # 不调整学习率
params = OrderedDict({
    "horizon":np.arange(1, 2,dtype=int).tolist(), #预测horizon
})
# informer, autoformer, transformer
# args.seq_len, args.label_len, args.pred_len = 80, 10, 20
# params = OrderedDict({
#     'model': ["autoformer", "informer", "transformer"],
#     'target':[args.file_name] + [f"{args.file_name}{i}" for i in range(3)]
# })

for args in product_args(args, params):
    logger.info(args)
    for target in [args.file_name]+[f"{args.file_name}{i}" for i in range(3)]:
        args.target = target
        for ii in range(args.itr):
            # setting record of experiments
            args.des = args.target
            setting_keys = '{}_{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'
            setting_values=[
                        args.model, args.dataset, args.features, 
                        args.seq_len, args.label_len, args.pred_len,
                        args.input_size, args.out_size, args.horizon,
                        args.batch_size, args.learning_rate,
                        args.des, "standar"]

            setting = setting_keys.format(*setting_values)
            params_dict = get_params_dict(setting_keys, setting_values, None)

            logger.info('Args in experiment:{}'.format(setting))
            # set experiments
            exp = Exp_model(args, setting, params_dict)
            
            if not args.do_predict:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(plot=False, save=True)
            else:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(plot=True, writer=False, save=True)

            torch.cuda.empty_cache()
            break
        break


