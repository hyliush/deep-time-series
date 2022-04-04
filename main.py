import torch
from utils import logger
from args import args
from exp.exp_main import Exp_model

# args.model = "autoformer"
# args.dataset = "Mydata"
# args.do_predict = False
# args.debug = True
# args.input_params = ["x"]
# args.horizon = 3
logger.info(args)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_is{}_os{}_hn{}_bs{}_lr{}_{}_{}'.format(
                args.model, args.dataset, args.features, 
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
        exp.test(load=True)

    torch.cuda.empty_cache()
    break


