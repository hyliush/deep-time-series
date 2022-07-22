import torch
from torch import nn
import torch.nn.functional as F
from utils.activation import SoftRelu
from layers.Decompose import series_decomp
# explaination https://zhuanlan.zhihu.com/p/59172441?from_voters_page=true

class BaseTPA(nn.Module):
    def __init__(self):
        super().__init__()
    def main_forward(self, x):
        px = F.relu(self.input_proj(x))
        hs, (ht, _) = self.lstm(px) # hs 最后一层，所有步， batch_size * seq_len * hidden_size
        # ht = ht.view(-1, 2, ht.shape[-2], ht.shape[-1])
        # ht = ht[-1].view(ht.shape[-2], -1)
        ht = ht[-1] # 最后一层，最后一步的hidden_state, batch_size * hidden_size
        final_h = self.att(hs, ht)  # 最后一步的ht'， fig2
        return final_h.unsqueeze(1)

class TPA_Attention(nn.Module):
    def __init__(self, seq_len, tpa_hidden_size):
        super(TPA_Attention, self).__init__()
        self.n_filters = 32 # out_channels
        self.filter_size = 1 
        self.conv = nn.Conv2d(1, self.n_filters, (seq_len, self.filter_size)) # kernel_size=(seq_len*1)，本质为conv1d，特征维度上移动
        self.Wa = nn.Parameter(torch.rand(self.n_filters, tpa_hidden_size))
        self.Whv = nn.Linear(self.n_filters+tpa_hidden_size, tpa_hidden_size)

    def forward(self, hs, ht):
        '''
        Args:
            ht: 最后一层，最后一步的hidden_state, B*hidden_size
            hs: 最后一层，每一步的lstm_out, B * seq_len *hidden_size

        '''
        hs = hs.unsqueeze(1) # B * 1 * seq_len *hidden_size
        H = self.conv(hs)[:, :, 0]  # B x n_filters x hidden_size
        H = H.transpose(1, 2) # fig2 H
        alpha = torch.sigmoid(torch.sum((H @ self.Wa) * ht.unsqueeze(-1), dim=-1))  # B x hidden_size
        V = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # B x n_filters
        vh = torch.cat([V, ht], dim=1)
        return self.Whv(vh)


class TPA(BaseTPA):
    def __init__(self, args):
        
        '''
        Args:
            input_size: features dim
            tpa_ar_len: ar regression using last tpa_ar_len
            out_size: need to be predicted series, last out_size series
            default pred_len = 1 
        '''
        super(TPA, self).__init__()
        input_size, seq_len, tpa_hidden_size, tpa_n_layers, tpa_ar_len, out_size = \
                args.input_size,\
                args.seq_len,\
                args.tpa_hidden_size, \
                args.tpa_n_layers,\
                args.tpa_ar_len,\
                args.out_size
        self.ar_len = tpa_ar_len
        self.args = args
        self.input_proj = nn.Linear(input_size, tpa_hidden_size)
        self.lstm = nn.LSTM(input_size=tpa_hidden_size, hidden_size=tpa_hidden_size,
                            num_layers=tpa_n_layers, batch_first=True)
        self.att = TPA_Attention(seq_len, tpa_hidden_size)
        
        # self.out_proj = nn.Linear(tpa_hidden_size, out_size) #多变量预测，改为pred_len，则为多步预测

        # self.ar = nn.Linear(self.ar_len, args.pred_len) # 当预测多个序列时，实际上共享了AR参数了，一种解决办法，设置多个ar层分别处理不同序列

        if self.args.decompose:
            self.decomp = series_decomp(self.args.moving_avg)
            self.trend_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=tpa_hidden_size, 
                        kernel_size=3, stride=1, padding=1,
                        padding_mode='circular', bias=False),
                nn.Linear(args.seq_len, args.pred_len),
                # nn.GELU()
                        )
        if self.args.criterion == "gaussian":
            self.n_params = 2
            self.activation = SoftRelu()
        elif self.args.criterion == "quantile":
            self.n_params = 3
        else:
            self.n_params = 1
        self.output_layer = nn.Linear(tpa_hidden_size, self.n_params*self.args.out_size)

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
        if self.args.criterion == "gaussian":
            output[...,-1] = self.activation(output[...,-1])
        if self.args.criterion == "mse":
            output = output.squeeze(dim=-1)
        return output
