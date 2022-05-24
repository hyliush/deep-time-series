import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model
from utils.search import get_args
from utils.tools import get_params_dict
from collections import OrderedDict
import numpy as np

# args.model = "tcn"
# args.load = True
# args.seq_len, args.label_len, args.pred_len = 144, 36, 288
# args.output_attention = False
# args.des = "test_tmpdata"
# args.file_name = "Mydata.csv"
args.model = "transformer"
args.seq_len, args.label_len, args.pred_len = 80, 20, 40
args.dataset = "google"
# args.horizon = 1
# args.input_params = ["x"]
# args.learning_rate = 0.001
# args.lradj = "type10"
params = OrderedDict({
    "horizon":np.arange(1, 41,dtype=int).tolist()
})
params = None
args.patience = 3
args.do_predict = True
# args.out_inverse = True
# args.debug = True
# args.distil = False
# params = OrderedDict({
#     "use_conv":[True, False],
#     "use_bias":[True, False],
#     "use_aff":[True, False],
# })
# args.qk_size = args.qk_size//args.n_heads
# args.uv_size = args.uv_size//args.n_heads
# args.debug = False
# args.activation = "swish"
# args.des = "addp"
# args.do_predict = True
for args in get_args(args, params):
    logger.info(args)
    for ii in range(args.itr):
        # setting record of experiments
        setting_keys = '{}_{}_ty{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'
        setting_values=[
                    args.model, args.dataset, args.test_year, args.features, 
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
            exp.test(plot=False)
        else:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(load=args.load, plot=False, writer=False, save=True)

        torch.cuda.empty_cache()
        break


