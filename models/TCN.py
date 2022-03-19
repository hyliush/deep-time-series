# Modified from https://github.com/locuslab/TCN
# conv1d conv2d https://blog.csdn.net/liujh845633242/article/details/102668515

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
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
        self.out_proj = nn.Linear(tcn_hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: batch_size * seq_len, input_size
        '''
        x = x.transpose(1, 2)
        out = self.network(x)[:, :, -1:] # 最后一步
        out = self.out_proj(out.transpose(1, 2))
        return out

if __name__ == '__main__':
    import torch
    batch_size, seq_len, input_size = 32, 10, 45
    x = torch.randn((batch_size, seq_len, input_size))
    out_size = 1
    tcn_hidden_size = 100
    tcn_num_layers = 3
    drop_out = 0.2
    tcn = TCN(input_size, tcn_hidden_size, tcn_num_layers, drop_out, out_size)
    tcn(x)

    print("end")