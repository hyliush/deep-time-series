import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model

# args.model = "autoformer"
# args.dataset = "ETTh1"
# args.data = "ETTh1"

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_{}_{}'.format(
                args.model, args.dataset, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.input_size, args.out_size,
                args.des, ii)
    logger.info('Args in experiment:{}'.format(setting))
    # set experiments
    exp = Exp_model(args)
    
    train = False
    if train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load=True)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, True)

    torch.cuda.empty_cache()
    break


