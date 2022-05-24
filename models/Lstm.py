import torch.nn as nn
import torch
class BenchmarkLstm(nn.Module):
    """Example network for solving Oze datachallenge.

    Attributes
    ----------
    lstm: Torch LSTM
        LSTM layers.
    linear: Torch Linear
        Fully connected layer.
    """

    def __init__(self, args):
        """Defines LSTM and Linear layers.

        Parameters
        ----------
        input_size: int, optional
            Input dimension. Default is 45 (features_dim).
        hidden_size: int, optional
            Latent dimension. Default is 100.
        out_size: int, optional
            Output dimension. Default is 1.
        num_layers: int, optional
            Number of LSTM layers. Default is 3.
        """
        super().__init__()
        self.args = args
        input_size, hidden_size, out_size, num_layers = args.input_size, args.lstm_hidden_size, args.out_size, args.lstm_n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, out_size)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        """Propagate input through the network.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape (batchsize, seq_len, features_dim)

        Returns
        -------
        output: Tensor
            Output tensor with shape (batchsize, *seq_len, out_size)
        """
        if self.args.importance:

            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.transpose(1, 2)
            x = x.to(torch.device("cuda"))
        lstm_out, _ = self.lstm(x.float())
        lstm_out = self.dropout(lstm_out[:,-1:,:])
        output = self.linear(lstm_out)
        return self.activation(output)
