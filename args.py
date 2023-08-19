import argparse
import torch
parser = argparse.ArgumentParser(description='Time Series Forecasting')

#  ------------------------- DATA AND TASK--------------------------------
# basic
parser.add_argument('--model', type=str, default='autoformer',help='model of experiment, options: [lstm, \
mlp, tpa, tcn, trans, gated, informerstack, informerlight(TBD)], autoformer, transformer,\
edlstm, edgru, edgruattention')
parser.add_argument('--data', type=str, default='default',
                    help='A dict for revising some params related to the data. AlL related arguments can be set by the `data[dict]`. See `data_parser` below')

#  arguments, which decide how to use data, can be set in utils.data.dataset.py manually.
#  最重要是写好dataset.py里的类，确定如何加载数据__getitem__，同时要生成self.train_idx,self.test_idx,self.val_idx
#  至于下面这些参数如何写并不重要
parser.add_argument('--dataset', type=str, default='Mydata', help='chose a dataset class to load data.(See utils.constants.py)')
parser.add_argument('--data_path', type=str, default='./data/Mydata/', help='root path of the data file')
parser.add_argument('--file_name', type=str, default='Mydata_partrv_softmax20.csv', help='file_name')
# if features == "MS" or "S", need to set `target1 and the predicted label must be the last column.
parser.add_argument('--target', type=str, default='rv', help='target feature in S task')
parser.add_argument('--start_col', type=int, default=1, help='Index of the start column of the continuous variables')
parser.add_argument('--date_col', type=str, default="date", help='date column name')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--inverse', action='store_true', help='inverse output data')
parser.add_argument('--scale', default=True, type=bool, help='scale input data')
parser.add_argument('--out_inverse', action='store_true', help='inverse output data')
parser.add_argument('--horizon', type=int, default=1, help='predict timeseries horizon-th in head.When many2many, means from 1(default) to pred_len')

#  ------------------------- TRAIN --------------------------------
parser.add_argument('--criterion', type=str, default='mse', help='loss function')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--des', type=str, default='test',help='exp description')

#  ------------------------- MODEL SETTING--------------------------------
# model common
parser.add_argument('--out_size', type=int, default=1, help='output features size')
parser.add_argument('--seq_len', type=int, default=60, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')
parser.add_argument('--input_params', type=str, nargs="+", default=["x", 'x_mark', 'y', 'y_mark'], help='input_params')
parser.add_argument('--target_param', type=str, default="y", help='target_params')
parser.add_argument('--test_year', type=int, default=2017, help='test year')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# seq2seq common
parser.add_argument('--enc_in', type=int, default=37, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='teacher_forcing_ratio')

## informer, autoformer, transformer
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--uv_size', type=int, default=2048, help='dimension of uv')
parser.add_argument('--qk_size', type=int, default=512, help='dimension of qk')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--enc_attn', type=str, default='prob', help='attention used in encoder(only in informer), options:[gate, prob, full]')
parser.add_argument('--dec_selfattn', type=str, default='full', help='selfattention used in encoder(in gau), options:[gate, prob, full]')
parser.add_argument('--dec_crossattn', type=str, default='full', help='crossattention used in encoder(in gau), options:[gate, prob, full]')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--use_conv', action='store_true', help='use conv1d attention in GateAtten')
parser.add_argument('--use_bias', action='store_true', help='use bias in GateAtten')
parser.add_argument('--use_aff', action='store_true', help='use aff in GateAtten')
parser.add_argument('--decompose', action='store_true', help='season decompose')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# nonseq2seq comon
parser.add_argument('--input_size', type=int, default=300, help='input features dim')
## LSTNet
parser.add_argument('--hidRNN', default=100, help='RNN hidden zie')
parser.add_argument('--hidCNN', type=int, default=100, help='CNN hidden size')
parser.add_argument('--hidSkip', type=float, default=5)
parser.add_argument('--skip', type=int, default=20)
parser.add_argument('--CNN_kernel', default=6, help='kernel size')
parser.add_argument('--highway_window', type=float, default=20, help='ar regression used last * items')

## tcn
parser.add_argument('--tcn_n_layers', default=3, help='num_layers')
parser.add_argument('--tcn_hidden_size', type=int, default=64, help='tcn hidden size')
parser.add_argument('--tcn_dropout', type=float, default=0.05, help='dropout')

## tpa
parser.add_argument('--tpa_n_layers', default=3, help='num_layers')
parser.add_argument('--tpa_hidden_size', type=int, default=64, help='tpa hidden size')
parser.add_argument('--tpa_ar_len', type=int, default=10, help='ar regression used last * items')

## trans
parser.add_argument('--trans_n_layers', default=3, help='num_layers')
parser.add_argument('--trans_hidden_size', type=int, default=256, help='trans hidden size')
parser.add_argument('--trans_n_heads', type=int, default=8, help='num of attention heads')
parser.add_argument('--trans_kernel_size', type=int, default=6, help='output size')

## lstm
parser.add_argument('--lstm_n_layers', type=int, default=2)
parser.add_argument('--lstm_hidden_size', type=int, default=64)

## mlp
parser.add_argument('--mlp_hidden_size', type=int, default=64)

## gated
parser.add_argument('--n_spatial', type=int, default=154, help='num of spatial')
parser.add_argument('--gdnn_embed_size', type=int, default=512, help='gdnn_embed_size')
parser.add_argument('--gdnn_hidden_size1', type=int, default=150, help='lstm hidden size')
parser.add_argument('--gdnn_hidden_size2', type=int, default=50, help=' combined model hidden size')
parser.add_argument('--gdnn_out_size', type=int, default=100, help='lstm output size')

# deepar
parser.add_argument('--data_folder', default='../timeseries-data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--load', default=False, help='load last trained model')

parser.add_argument('--val_num', type=int, default=1, help='val_num in one epoch')
parser.add_argument('--single_file', type=bool, default=True, help='single_file')
parser.add_argument('--debug', action='store_true', help='whether debug')
parser.add_argument('--importance', type=bool, default=False, help='importance')
parser.add_argument('--tqdm', type=bool, default=True, help='tqdm')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data_path':'./data/ETT/', 'file_name':'ETTh1.csv',"dataset":"ETTh1",
    "freq":'h', "features":"MS", 'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], "freq":'t'},
    'ETTm2':{'T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'Mydata':{'freq':'b', 'T':'rv',"features":"MS"},
    'google':{"features":"S", 'MS':[2,2,1],'M':[2,2,2], "S":[1,1,1]},
    "SDWPF":{'freq':'10min', 'T':'Patv',"features":"MS", 'MS':[10,10,1],'M':[10,10,10]},
    'Toy':{'data_path':'./data/ToyData', 'seq_len':96, 'label_len':0, "pred_len":24, "MS":[1,1,1], "T":"s"},
    'oze':{'seq_len':672, 'label_len':1, "pred_len":671, "M":[37,8,8], "T":"s", 'features':"M"},
    "oil":{'data_path':'./data/oil', "file_name":'wtimulti.csv', "dataset":"oil",
    "features":"MS", "T":"wti", "MS":[3,3,1],"S":[1, 1, 1]},
    "exchange":{'data_path':'./data/exchange_rate', "file_name":'exchange_rate.csv', "dataset":"exchange",
    "features":"MS", "T":"OT", "MS":[8,8,1],"S":[1, 1, 1]}
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.features = data_info.get("features") or args.features
    args.enc_in, args.dec_in, args.out_size = data_info.get(args.features)
    args.data_path = data_info.get("data_path") or args.data_path
    args.file_name = data_info.get("file_name") or args.file_name
    args.dataset = data_info.get("dataset") or args.dataset
    args.horizon = data_info.get("horizon") or args.horizon
    args.single_file = data_info.get("single_file", args.single_file)
    args.seq_len = data_info.get('seq_len') or args.seq_len
    args.label_len = data_info.get('label_len') or args.label_len
    args.pred_len = data_info.get('pred_len') or args.pred_len
    args.target = data_info.get("T") or args.target
    args.freq = data_info.get("freq") or args.freq
    args.date_col = data_info.get("date_col", args.date_col)
args.input_size = args.enc_in