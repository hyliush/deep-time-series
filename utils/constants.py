from models.seq2seq import Informer, Autoformer, Transformer, GruAttention, Gru, Lstm
from models.Gdnn import Gdnn
from models.TCN import TCN
from models.TPA import TPA
from models.Trans import Trans
from models.DeepAR import DeepAR
from models.Lstm import BenchmarkLstm
from models.Mlp import BenchmarkMlp
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, ToyDatasetSeq2Seq, UbiquantDataSetNoraml, VolatilityDataSet, ToyDataset

dataset_dict = {
'ETTh1':Dataset_ETT_hour,
'ETTh2':Dataset_ETT_hour,
'ETTm1':Dataset_ETT_minute,
'ETTm2':Dataset_ETT_minute,
'WTH':Dataset_Custom,
'ECL':Dataset_Custom,
'Solar':Dataset_Custom,
'custom':Dataset_Custom,
'Volatility':VolatilityDataSet,
'Ubiquant':UbiquantDataSetNoraml,
'Toy': ToyDataset,
'ToySeq2Seq': ToyDatasetSeq2Seq
}

model_dict = {
'edlstm': Lstm,
'edgru': Gru,
'edgruattention':GruAttention,
'informer':Informer,
'transformer': Transformer,
'autoformer': Autoformer,
'mlp':BenchmarkMlp,
'lstm':BenchmarkLstm,
'tcn':TCN,
'tpa':TPA,
'trans':Trans,
'gated':Gdnn,
'deepar':DeepAR
}