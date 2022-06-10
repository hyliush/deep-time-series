import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import get_args
from utils.tools import get_params_dict
from collections import OrderedDict
import numpy as np
# LSTM
# args.model = "lstm"
# args.seq_len, args.label_len, args.pred_len = 80, 0, 1
# args.dataset = "google1"
# args.input_params = ["x"]
# args.learning_rate = 0.001
# args.lradj = "type10"
# params = OrderedDict({
#     "horizon":np.arange(1, 21,dtype=int).tolist(),
#     'target':["mean CPU usage rate","mean CPU usage rate0","mean CPU usage rate1", "mean CPU usage rate2"]
# })

# informer, autoformer, transformer
args.seq_len, args.label_len, args.pred_len = 80, 10, 20
args.dataset = "google1"
params = OrderedDict({
    'model': ["informer", "transformer", "autoformer"],
    'target':["mean CPU usage rate","mean CPU usage rate0", "mean CPU usage rate1", "mean CPU usage rate2"]
})

args.patience = 3
args.do_predict = False
args.out_inverse = True

for args in get_args(args, params):
    logger.info(args)
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
            exp.test(load=args.load, plot=True, writer=False, save=True)

        torch.cuda.empty_cache()
        break


