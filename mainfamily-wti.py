import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import product_args
from utils.tools import get_params_dict
from collections import OrderedDict
import numpy as np

args.patience = 3 # earlystop
args.do_predict = False
args.out_inverse = True
args.debug = False
args.tqdm = False

args.input_params = ["x"] #指定模型的输入
args.learning_rate = 0.001
args.lradj = "type10" # 不调整学习率
args.date_col = ""
# args.data_path = "./data/oil"
args.data_path = "/kaggle/input/wtimulti"
args.target = "wti"
args.seq_len, args.label_len, args.pred_len = 60, 0, 1
args.cols = ["date", "wti", "DJIA_vix", "Three_Component_Index", "TWEXB"]
HORIZON = 1
params = OrderedDict({
    "dataset":["wtimulti"],
    "horizon":np.arange(1, HORIZON+10,dtype=int).tolist(), #预测horizon
    "model":["trans", "tpa", "lstm", "tcn"],
    "decompose":[True, False],
    "features":["S"],
    "criterion": ["gaussian", "quantile", "mse"]
})


# informer, autoformer, transformer
# args.seq_len, args.label_len, args.pred_len = 96, 48, 24
# params = OrderedDict({
#     'model': ["autoformer"],
#     # "model": ["transformerdecompose", "transformer", "informer", "informerdecompose", "autoformer"],
# })

for args in product_args(args, params):
    logger.info(args)
    for ii in range(args.itr):
        if args.features == "MS":
            args.enc_in, args.dec_in, args.out_size = len(args.cols)-1, len(args.cols)-1, 1
        if args.features == "S":
            args.enc_in, args.dec_in, args.out_size = 1, 1, 1
        args.input_size = args.enc_in
        # setting record of experiments
        args.file_name = args.dataset+".csv"
        model =  f"{args.model}decompose" if args.decompose else f"{args.model}"
        setting_keys = '{}_{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_ct{}_{}_{}'
        setting_values=[
                    model, args.dataset, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.input_size, args.out_size, args.horizon,
                    args.batch_size, args.learning_rate,args.criterion,
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
            exp.test(plot=False, writer=False, save=False)

        torch.cuda.empty_cache()
        
        break


