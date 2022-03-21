import torch
from utils import logger
from args import args

data_parser = {
    'ETTh1':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'Volatility':{'data_path':'D:/News_topics/RV-predictability/save_file/volatility_tmp',
    'freq':'b', 'T':'rv','M':[45,45,45],'S':[1,1,1],'MS':[45,45,1],
    'seq_len':10, 'label_len':1, "pred_len":1},
    'Ubiquant':{'data_path':'D:/IDEA/Spatial-temporal/ubiquant/ubiquantSeg',
    'freq':'b', 'T':'target','M':[45,45,45],'S':[1,1,1],'MS':[45,45,1],
    'seq_len':25, 'label_len':0, "pred_len":1},
    'Toy':{'seq_len':96, 'label_len':0, "pred_len":24, "MS":[1,1,1], "T":"s"}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    if "data_path" in data_info:
        args.data_path = data_info["data_path"]
    if 'freq' in data_info:
        args.freq = data_info["freq"]
    if 'seq_len' in data_info:
        args.seq_len, args.label_len, args.pred_len = data_info['seq_len'], data_info['label_len'], data_info['pred_len']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.out_size = data_info[args.features]
    args.input_size = args.enc_in
args.detail_freq = args.freq
args.freq = args.freq[-1:]

from exp.exp_main import Exp_model
args.model = "edgru"
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}__sl{}_ll{}_pl{}_id{}_co{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.input_size, args.out_size,
                args.des, ii)
    logger.info('Args in experiment:{}'.format(setting))
    # set experiments
    exp = Exp_model(args)

    train = True
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
        exp.predict(setting, True)

    torch.cuda.empty_cache()
    break


