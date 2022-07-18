import torch.nn as nn
import torch
from utils.activation import SoftRelu

class AR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_size, hidden_size = args.input_size, args.mlp_hidden_size
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
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
        self.output_layer = nn.Linear(hidden_size, self.n_params*self.args.out_size)

    def forward(self, x):
        if self.args.importance:
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x.to(torch.device("cuda"))

        out = self.trend_layer(x.transpose(1, 2)).transpose(1, 2)
        output = self.output_layer(out).view(-1, self.args.pred_len, self.args.out_size, self.n_params)

        if self.args.criterion == "guassian":
            output[...,-1] = self.activation(output[...,-1])
        if self.args.criterion == "mse":
            output = output.squeeze(dim=-1)
        return output