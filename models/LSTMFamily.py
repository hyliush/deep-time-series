from torch import nn as nn
import torch
from layers.Decompose import series_decomp
from utils.activation import SoftRelu

class BaseLSTM(nn.Module):
    def __init__(self):
        super().__init__()

    def main_forward(self, x):
        lstm_out, _ = self.lstm(x.float())
        lstm_out = self.dropout(lstm_out[:,-1:,:])
        return lstm_out

class LSTM(BaseLSTM):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_size, hidden_size, out_size, num_layers = args.input_size, args.lstm_hidden_size, args.out_size, args.lstm_n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        if self.args.decompose:
            self.decomp = series_decomp(self.args.moving_avg)
            self.trend_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
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
        self.output_layer = nn.Linear(hidden_size, self.n_params*self.args.out_size)
        # self.output_layer = nn.Linear(hidden_size, self.args.out_size)
    def forward(self, x):
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x.to(torch.device("cuda"))

        if self.args.decompose:
            season, trend = self.decomp(x)
            season_out = self.main_forward(season)
            # season_out = self.output_layer(season_out)
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