import torch.nn as nn
import torch
from layers.Decompose import series_decomp
class LSTMDecompose(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.args = args
        input_size, hidden_size, out_size, num_layers = args.input_size, args.lstm_hidden_size, args.out_size, args.lstm_n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, out_size)
        self.activation = nn.LeakyReLU(0.1)

        moving_avg = 25
        self.decomp = series_decomp(moving_avg)
        self.projection = nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3, stride=1, padding=1,
                    padding_mode='circular', bias=False)
        self.out_proj2 = nn.Linear(args.seq_len, args.pred_len)

    def forward(self, x):
        if self.args.importance:

            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x.to(torch.device("cuda"))

        x, trend = self.decomp(x)
        lstm_out, _ = self.lstm(x.float())
        lstm_out = self.dropout(lstm_out[:,-1:,:])
        output = self.linear(lstm_out)

        residual_trend = self.projection(trend.permute(0, 2, 1))
        residual_trend = self.out_proj2(residual_trend).transpose(1, 2)
        
        return self.activation(output)+residual_trend
