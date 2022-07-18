import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import product_args
from utils.tools import get_params_dict
from collections import OrderedDict
import numpy as np
args.dataset = "oil"
args.patience = 5 # earlystop
args.do_predict = False 
args.out_inverse = True
# args.scale = False
args.val_num = 1
args.data_path = "./data/oil"
# args.data_path = "./data"

args.seq_len, args.label_len, args.pred_len = 60, 0, 1
args.input_params = ["x"] #指定模型的输入
args.learning_rate = 0.001
args.lradj = "type10" # 不调整学习率
args.date_col = ""
HORIZON = 1
params = OrderedDict({
    "file_name": ["wti"],
    "horizon":np.arange(1, HORIZON+1,dtype=int).tolist(), #预测horizon
    "model":["lstm", "lstmdecompose", "tpa","tpadecompose", "trans", "transdecompose", "tcn", "tcndecompose"],
    # "model":["mlpdecompose"],
    # "model":["lstmdecompose", "transdecompose", "tpadecompose", "tcndecompose"],
    # "file_name": ["mean CPU usage rate", "5f5533cpu", "53ea38cpu", "assigned memory usage"],
})

# informer, autoformer, transformer
# args.date_col = "Date"
# args.seq_len, args.label_len, args.pred_len = 60, 30, 10
# params = OrderedDict({
#     "file_name":["wti"],
#     'model': ["informer", "informerdecompose", "transformer", "transformerdecompose"],
#     # "file_name": ["mean CPU usage rate","5f5533cpu", "53ea38cpu", "assigned memory usage"]
# })
N = 3
for args in product_args(args, params):
    logger.info(args)
    target_lst =  [args.file_name]+[f"{args.file_name}{i}" for i in range(N)]
    args.file_name = args.file_name+".csv"
    for target in target_lst:
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
                        args.des, ii]

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
                exp.test(plot=False, writer=False, save=True)

            torch.cuda.empty_cache()
        break
        if "decompose" in args.model:
            break


