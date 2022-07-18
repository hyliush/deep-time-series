import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import product_args
from utils.tools import get_params_dict, get_cols
from collections import OrderedDict
import numpy as np

# model
# 提前把enc_dim 注释掉，更改默认data配置，更改Single的test函数
args.seq_len, args.label_len, args.pred_len = 40, 0, 1
args.input_params = ["x"]
args.learning_rate = 0.001
args.lradj = "type10"

params = OrderedDict({
    'model': ["lstm"], # "tcn", "tpa", 
    "horizon":np.arange(1, 2, dtype=int).tolist(),
    "des":["base", "market", "industry", "cos", "ret"],
    "test_year":np.arange(2017, 2022, dtype=int).tolist(),
    
})
# params = OrderedDict({
#     'model': ["lstm"],
#     "horizon":np.arange(1, 2, dtype=int).tolist(),
#     "des":["base"],
#     "test_year":np.arange(2017, 2022, dtype=int).tolist(),
# })
args.debug = True
args.do_predict = False
args.patience = 2
args.out_inverse = True
args.val_num = 4

# informer, autoformer, transformer
# args.seq_len, args.label_len, args.pred_len = 80, 10, 20
# params = OrderedDict({
#     'model': ["autoformer", "informer", "transformer"],
#     'target':[args.file_name] + [f"{args.file_name}{i}" for i in range(3)]
# })
ii = 0
for args in product_args(args, params):
    for i in range(args.itr):
        # args.cols = get_cols(des=args.des)
        # args.des = f"{args.des}"
        if i == 0:
            args.cols = get_cols(des=args.des)
            args.des = f"{args.des}"
            continue
        else:
            args.cols = get_cols(des=args.des, notw=True, notm=True)
            args.des = f"{args.des}{1}{1}"
        
        tmplen = len(args.cols) - 1
        args.enc_in, args.dec_in, args.input_size, args.out_size = \
        tmplen, tmplen, tmplen, 1

        # setting record of experiments
        setting_keys = '{}_{}_ft{}_ty{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'
        setting_values=[
                    args.model, args.dataset, args.features, args.test_year,
                    args.seq_len, args.label_len, args.pred_len,
                    args.input_size, args.out_size, args.horizon,
                    args.batch_size, args.learning_rate,
                    args.des, "span"]

        # setting_keys = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_ea{}_da1{}_da2{}_fc{}_eb{}_dt{}_mx{}_uconv{}_ubias{}_uaff{}_ac1{}_ac2{}_uv{}_qk{}_{}_{}'
        # setting_values = [args.model, args.dataset, args.features, 
        #         args.seq_len, args.label_len, args.pred_len,
        #         args.d_model, args.n_heads, args.e_layers, args.d_layers, 
        #         args.d_ff, args.enc_attn, args.dec_selfattn, args.dec_crossattn,
        #         args.factor, args.embed, int(args.distil), int(args.mix),
        #         int(args.use_conv), int(args.use_bias), int(args.use_aff),
        #         args.activation, args.test_activation,
        #         args.uv_size, args.qk_size, args.des, ii]

        setting = setting_keys.format(*setting_values)
        params_dict = get_params_dict(setting_keys, setting_values, None)

        logger.info(args)
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
            exp.test(plot=True, writer=True, save=False)

        torch.cuda.empty_cache()


