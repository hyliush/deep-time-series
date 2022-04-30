import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model

# args.model = "gau"
# args.do_predict = True
# args.debug = True
# args.load = True
# args.seq_len, args.label_len, args.pred_len = 96, 48, 24
# args.input_params = ["x"]
# args.horizon = 1
# args.seq_len, args.label_len, args.pred_len = 144, 36, 288
# args.output_attention = False
# args.des = "test_tmpdata"
# args.file_name = "tmpMydata.csv"
# args.data_path = "./data/SDWPF"
# args.file_name = "sdwpf_baidukddcup2022_turb1.csv"
# args.dataset = "SDWPF"

logger.info(args)
for ii in range(args.itr):
    # setting record of experiments
    # setting = '{}_{}_ty{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'.format(
    #             args.model, args.dataset, args.test_year, args.features, 
    #             args.seq_len, args.label_len, args.pred_len,
    #             args.input_size, args.out_size, args.horizon,
    #             args.batch_size, args.learning_rate,
    #             args.des, ii)

    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_ea{}_da1{}_da2{}_fc{}_eb{}_dt{}_mx{}_uconv{}_ubias{}_uv{}_qk{}_{}_{}'.format(
                args.model, args.dataset, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, 
                args.d_ff, args.enc_attn, args.dec_selfattn, args.dec_crossattn, args.factor, 
                args.embed, int(args.distil), int(args.mix),
                int(args.use_conv), int(args.use_bias),
                args.uv_size, args.qk_size,
                args.des, ii)

    logger.info('Args in experiment:{}'.format(setting))
    # set experiments
    exp = Exp_model(args, setting)
    
    if not args.do_predict:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test()
    else:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(load=args.load)

    torch.cuda.empty_cache()
    break


