# Modified from https://github.com/locuslab/TCN
# conv1d conv2d https://blog.csdn.net/liujh845633242/article/details/102668515

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from layers.Decompose import series_decomp
from utils.activation import SoftRelu

class Chomp1d(nn.Module):
    '''
    Args:
        remove padding
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class BaseTCN(nn.Module):
    def __init__(self):
        super().__init__()

    def main_forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)[:, :, -1:].transpose(1, 2) # 最后一步
        return out

class TCN(BaseTCN):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.args = args
        input_size, tcn_hidden_size, tcn_n_layers, tcn_dropout, out_size = \
                args.input_size,\
                args.tcn_hidden_size, \
                args.tcn_n_layers,\
                args.tcn_dropout,\
                args.out_size
        num_channels = [tcn_hidden_size] * tcn_n_layers
        kernel_size = 2
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # one temporalBlock can be seen from fig1(b).
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=tcn_dropout)]

        self.network = nn.Sequential(*layers)

        if self.args.decompose:
            self.decomp = series_decomp(self.args.moving_avg)
            self.trend_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=tcn_hidden_size, 
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
        self.output_layer = nn.Linear(tcn_hidden_size, self.n_params*self.args.out_size)
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
