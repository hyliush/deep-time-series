import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activation import SoftRelu
from layers.Decompose import series_decomp

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BaseTrans(nn.Module):
    def __init__(self):
        super().__init__()
    def main_forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size-1,0))
        x = self.conv(x).permute(2, 0, 1)
        x = self.pos_encoder(x)
        out = self.transformer(x).transpose(0, 1)[:, -1:]
        return out

class Trans(BaseTrans):
    def __init__(self, args):
        super(Trans, self).__init__()
        input_size = args.input_size
        trans_hidden_size = args.trans_hidden_size
        trans_kernel_size = args.trans_kernel_size
        seq_len = args.seq_len
        n_trans_head = args.trans_n_heads
        trans_n_layers = args.trans_n_layers
        out_size = args.out_size
        self.kernel_size = trans_kernel_size

        self.conv = nn.Conv1d(input_size, trans_hidden_size, kernel_size=trans_kernel_size)
        self.pos_encoder = PositionalEncoding(trans_hidden_size, max_len=seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=trans_hidden_size, nhead=n_trans_head)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=trans_n_layers)

        if self.args.decompose:
            self.decomp = series_decomp(self.args.moving_avg)
            self.trend_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=trans_hidden_size, 
                        kernel_size=3, stride=1, padding=1,
                        padding_mode='circular', bias=False),
                nn.Linear(args.seq_len, args.pred_len),
                nn.GELU()
                        )
        if self.args.criterion == "gaussian":
            self.n_params = 2
            self.activation = SoftRelu()
        elif self.args.criterion == "quantile":
            self.n_params = 3
        else:
            self.n_params = 1
        self.output_layer = nn.Linear(trans_hidden_size, self.n_params*self.args.out_size)

    def forward(self, x):
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x.to(torch.device("cuda"))

        if self.args.decompose:
            season, trend = self.decomp(x)
            season_out = self.main_forward(season)
            trend_out = self.trend_layer(trend.transpose(1, 2)).transpose(1, 2)
            out = trend_out + season_out
        else:
            out = self.main_forward(x)
            
        output = self.output_layer(out).view(-1, self.args.pred_len, self.args.out_size, self.n_params)
        if self.args.criterion == "guassian":
            output[...,-1] = self.activation(output[...,-1])
        if self.args.criterion == "mse":
            output = output.squeeze(dim=-1)
        return output