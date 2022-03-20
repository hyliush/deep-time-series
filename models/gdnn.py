import torch.nn as nn
from utils.activation import Swish
from models.informer.embed import SpatialEmbedding, TemporalEmbedding, TokenEmbedding, FixedEmbedding

class Gdnn(nn.Module):
    def __init__(self, n_spatial=146, gdnn_embed_size=512, embed_type='fixed', freq='h',
                input_size=45, gdnn_hidden_size1=150, gdnn_out_size=100, num_layers=3,
                gdnn_hidden_size2=50, out_size=1):
        '''
        Args:
            n_spatial): num of spatial
            gdnn_embed_size (int): embedding dimension
            gdnn_hidden_size1 (int): lstm hidden dimension
            gdnn_out_size (int): lstm output dimension
            input_size (int): features dimension
            out_size (int): forescast dimension
            gdnn_hidden_size2 (int): combined net hidden dimension

        '''
        super(Gdnn, self).__init__()
        # st_net
        self.st_net = St_net(n_spatial, gdnn_embed_size, embed_type, freq)

        # lstm
        self.lstm1 = Lstm(input_size, gdnn_hidden_size1, gdnn_out_size, num_layers)
        self.lstm2 = Lstm(gdnn_embed_size, gdnn_hidden_size1, gdnn_out_size, num_layers)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()

        # conbined
        self.linear1 = nn.Linear(gdnn_out_size, gdnn_hidden_size2)
        self.linear2 = nn.Linear(gdnn_hidden_size2, out_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, x_temporal, x_spatial):
        # st_net
        x_st = self.st_net(x_temporal, x_spatial)

        # dnn_net
        lstm_out1 = self.swish(self.lstm1(x))
        lstm_out2 = self.sigmoid(self.lstm2(x_st))

        # conbined
        net_combine = self.dropout(lstm_out1 * lstm_out2)
        out1 = self.swish(self.linear1(net_combine))
        out = self.linear2(out1)

        return out
    
class St_net(nn.Module):
    def __init__(self, n_spatial, gdnn_embed_size, embed_type, freq) -> None:
        super().__init__()
        # st_net
        self.spa_embed = SpatialEmbedding(n_spatial, gdnn_embed_size)
        self.tmp_embed = TemporalEmbedding(gdnn_embed_size, embed_type, freq)
        self.swish = Swish()

    def forward(self, x_temporal, x_spatial):
        # st_net
        x_spa_embed = self.spa_embed(x_spatial).squeeze()
        x_tmp_embed = self.tmp_embed(x_temporal)
        x_st = self.swish(x_spa_embed) * self.swish(x_tmp_embed)

        return x_st

class Lstm(nn.Module):
    def __init__(self, input_size=37, hidden_size=100, out_size=8, num_layers=3, **kwargs):
        super().__init__(**kwargs)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.float())
        lstm_out = self.dropout(lstm_out[:,-1:,:])
        output = self.linear(lstm_out)
        return self.dropout(output)


if __name__ == '__main__':
    import torch
    batch_size, seq_len = 32, 10
    x = torch.randn((batch_size, seq_len, 45))
    # month
    x_month = torch.randint(0, 12, (batch_size, seq_len, 1))
    # day
    x_day = torch.randint(0, 30, (batch_size, seq_len, 1))
    # weekday
    x_weekday = torch.randint(0, 7, (batch_size, seq_len, 1))
    x_temporal = torch.cat([x_month, x_day, x_weekday], dim=-1)

    x_spatial = torch.randint(0, 100, (batch_size, seq_len, 1))
    # fix = FixedEmbedding(200, 128)
    # fix(x_spatial)
    gatednn = Gdnn(n_spatial=146, gdnn_embed_size=512, embed_type='fixed', freq='h',
                input_size=45, gdnn_hidden_size1=150, gdnn_out_size=100, num_layers=3,
                gdnn_hidden_size2=50, out_size=1)
    out = gatednn(x, x_temporal, x_spatial)
    
    print(out)