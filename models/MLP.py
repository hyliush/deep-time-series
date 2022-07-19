from turtle import forward
import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size, hidden_size, out_size = args.input_size, args.mlp_hidden_size, args.out_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, out_size),
            nn.LeakyReLU(0.1)
        )

    def forward(x):
        return self.net(x)