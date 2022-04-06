import torch
from torch import nn
import torch.nn.functional as F
# explaination https://zhuanlan.zhihu.com/p/59172441?from_voters_page=true


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


class TPA(nn.Module):
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
        self.target_pos = args.target_pos
        self.ar_len = tpa_ar_len

        self.input_proj = nn.Linear(input_size, tpa_hidden_size)
        self.lstm = nn.LSTM(input_size=tpa_hidden_size, hidden_size=tpa_hidden_size,
                            num_layers=tpa_n_layers, batch_first=True)
        self.att = TPA_Attention(seq_len, tpa_hidden_size)
        
        self.out_proj = nn.Linear(tpa_hidden_size, out_size) #多变量预测，改为pred_len，则为多步预测

        pred_len = 1
        self.ar = nn.Linear(self.ar_len, pred_len) # 当预测多个序列时，实际上共享了AR参数了，一种解决办法，设置多个ar层分别处理不同序列

    def forward(self, x):
        # batch_size, seq_len, input_size = x.size()
        px = F.relu(self.input_proj(x))
        hs, (ht, _) = self.lstm(px) # hs 最后一层，所有步， batch_size * seq_len * hidden_size
        ht = ht[-1] # 最后一层，最后一步的hidden_state, batch_size * hidden_size
        final_h = self.att(hs, ht)  # 最后一步的ht'， fig2
        ar_out = self.ar(x[:, -self.ar_len:, [self.target_pos]].transpose(1, 2))[:, :, 0]
        out = self.out_proj(final_h) + ar_out
        out = out.unsqueeze(1) # add timesereis dim 
        return out

if __name__ == '__main__':
    import torch
    batch_size, seq_len, input_size = 64, 10, 45
    x = torch.randn((batch_size, seq_len, input_size))
    tpa_ar_len, out_size = 5, 12
    tpa = TPA(input_size, seq_len, 100, 3, tpa_ar_len, out_size)
    tpa(x)

    print("end")