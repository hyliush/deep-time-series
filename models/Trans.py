import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Trans(nn.Module):

    def __init__(self, input_size, trans_hidden_size, trans_kernel_size, 
                 seq_len, n_trans_head, _trans_n_layers, out_size):
        super(Trans, self).__init__()

        self.conv = nn.Conv1d(input_size, trans_hidden_size, kernel_size=trans_kernel_size)
        self.pos_encoder = PositionalEncoding(trans_hidden_size, max_len=seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=trans_hidden_size, nhead=n_trans_head)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=_trans_n_layers)
        self.fc = nn.Linear(trans_hidden_size, out_size)

        self.kernel_size = trans_kernel_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size-1,0))
        x = self.conv(x).permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x).transpose(0, 1)[:, -1:]
        output = self.fc(x)
        return output

if __name__ == "__main__":
    import torch 
    batch_size, seq_len, input_size = 32, 10, 45
    x = torch.randn(batch_size, seq_len, input_size)
    input_size, trans_hidden_size, trans_kernel_size\
        , seq_len, n_trans_head, _trans_n_layers = input_size, 256, 6, seq_len, 8, 3
    out_size = 1
    trans = Trans(input_size, trans_hidden_size, trans_kernel_size, seq_len, n_trans_head, _trans_n_layers, out_size)
    
    out = trans(x)
    
    print(out)
