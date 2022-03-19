from turtle import forward
import torch.nn as nn

class BenchmarkMlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size, hidden_size, out_size = args.input_size, args.mlp_hidden_size, args.out_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        outputs = self.net(x)
        return outputs

if __name__ == '__main__':
    import torch
    input_size, hidden_size, out_size = 10, 20, 1
    input = torch.randn((32, 10, input_size))
    model = BenchmarkMlp(input_size, hidden_size, out_size)
    out = model(input)

    print(out)