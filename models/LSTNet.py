import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.P = args.seq_len
        self.m = args.input_size
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.conv1 = nn.Conv1d(self.m, self.hidC, self.Ck)
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = F.tanh
 
    def forward(self, x):
        batch_size = x.size(0)
        
        #CNN
        #c = x.view(-1, 1, self.P, self.m)
        c = x.transpose(1, 2)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        # c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view((batch_size, self.hidC, self.pt, self.skip))
            s = s.permute(2,0,3,1).contiguous()# pt * batch_size * skip * hidC
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
        res = res.unsqueeze(dim=1)
        # if (self.output):
        #     res = self.output(res)
        return res

if __name__ == '__main__':
    _dict = {"seq_len":60, "input_size":33, "hidRNN":100, "hidCNN":100, "hidSkip":5,
    "CNN_kernel":6, "skip":24, "highway_window":24, "dropout":0.1}
    args = namedtuple("args", _dict.keys())
    args = args._make(_dict.values())
    x = torch.randn(32, args.seq_len, args.input_size)
    model = LSTNet(args)
    out = model(x)
    print(out)