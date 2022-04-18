import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model

# args.model = "informer"
# args.do_predict = False
# args.debug = True
# #args.input_params = ["x"]
# args.horizon = 1
# args.seq_len, args.label_len, args.pred_len = 144, 36, 288
# args.output_attention = False
# args.des = "test_tmpdata"
# args.file_name = "tmpMydata.csv"
# args.data_path = "./data/SDWPF"
# args.file_name = "sdwpf_baidukddcup2022_full.CSV"
# args.dataset = "SDWPF"
logger.info(args)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ty{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'.format(
                args.model, args.dataset, args.test_year, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.input_size, args.out_size, args.horizon,
                args.batch_size, args.learning_rate,
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


