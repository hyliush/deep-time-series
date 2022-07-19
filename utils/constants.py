from models.MultiHeadGAU import MultiHeadGAU
from models.seq2seq import Informer, Autoformer, Transformer, GruAttention, Gru, Lstm, Gaformer, TransformerDecompose, InformerDecompose
from models import Gdnn, TCN, TPA, Trans, DeepAR, MLP, LSTNet, GAU, AU, TCNDecompose, TPADecompose, LSTMDecompose, TransDecompose, AR, LSTM
from utils.data import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, MyDataSet, ToyDataset, MyDataset1
from models.informer1.model import Informer as Infomer1
from models.seq2seq.MultiHeadGaformer import MultiHeadGaformer
from utils.loss import GaussianLoss, OZELoss, QuantileLoss,Normal_loss
from torch import nn
dataset_dict = {
'ETTh1':Dataset_ETT_hour,
'ETTh2':Dataset_ETT_hour,
'ETTm1':Dataset_ETT_minute,
'ETTm2':Dataset_ETT_minute,
'WTH':Dataset_Custom,
'ECL':Dataset_Custom,
'Solar':Dataset_Custom,
'custom':Dataset_Custom,
'Mydata':MyDataSet,
'Toy': ToyDataset,
"google1":Dataset_Custom,
# "oil":GoogleDataset1,
"mydata": MyDataset1
}

model_dict = {
'edlstm': Lstm,
'edgru': Gru,
'edgruattention':GruAttention,
'informer':Informer,
'informer1':Infomer1,
'transformer': Transformer,
'autoformer': Autoformer,
'mlp':MLP,
"lstm":LSTM,
'tcn':TCN,
'tpa':TPA,
'trans':Trans,
'lstnet':LSTNet,
'gated':Gdnn,
'deepar':DeepAR,
'gaformer':Gaformer,
'multigaformer':MultiHeadGaformer,
"gau":GAU,
"multigau":MultiHeadGAU,
"au":AU,
"tcndecompose":TCNDecompose,
"tpadecompose":TPADecompose,
"lstmdecompose":LSTMDecompose,
"transdecompose":TransDecompose,
"transformerdecompose":TransformerDecompose,
"informerdecompose":InformerDecompose,
}

criterion_dict = {
"mse":nn.MSELoss(),
"normal":Normal_loss,
"gaussian":GaussianLoss(),
"quantile":QuantileLoss(),
"oze":OZELoss(alpha=0.3)
}