import torch.nn as nn
import random
import torch
import sys
from layers.Embed import DataEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_in, emb_dim, hid_dim, n_layers, embed, freq, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = DataEmbedding(enc_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)

    def forward(self, x_enc):
        
        #x_enc = [batch size, x_enc len, n_features]
        
        embedded = self.embedding(x_enc)
        #embedded = [batch size, x_enc len, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [batch size, x_enc len hid, dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = DataEmbedding(dec_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, dec_in)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        #input = [batch size, 1, n_features]
        embedded = self.embedding(input)
        #embedded = [batch size, 1, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [batch size, seq len, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Lstm(nn.Module):
    def __init__(self, args):
        super().__init__()

        enc_in, dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout = \
        args.enc_in, args.dec_in, args.d_model, args.d_model, 3, \
        args.embed, args.freq, args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.out_size = args.out_size
        self.encoder = Encoder(enc_in, emb_dim, hid_dim, n_layers, embed, freq, dropout)
        self.decoder = Decoder(dec_in, emb_dim, hid_dim, n_layers, embed, freq, dropout)
        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, x_enc, x_dec):
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0
        #x_enc = [x_enc len, batch size, n_features]
        #x_dec = [x_dec len, batch size, n_features]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size, x_dec_len, dec_in = x_dec.shape
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, x_dec_len-1, dec_in).to(x_enc.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x_enc)
        
        #first input to the decoder is the <sos> tokens
        if isinstance(x_dec, list):
            input = [i[:, 0, :].unsqueeze(dim=1) for i in x_dec]
        else:
            input = x_dec[:, 0, :].unsqueeze(dim=1)
        for t in range(1, x_dec_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t-1, :] = output.squeeze(dim=1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            if isinstance(x_dec, list):
                input0 = x_dec[0][:, t, :].unsqueeze(dim=1) if teacher_force else output
                input = [input0, x_dec[1][:, t, :].unsqueeze(dim=1)]
            else:
                input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
        return outputs[:, :, -self.out_size:]

if __name__ == '__main__':

    enc_in, dec_in, emb_dim, hid_dim, n_layers = 45, 45, 512, 64, 2
    batch_size, seq_len = 32, 10
    x = torch.randn(batch_size, seq_len, enc_in)
    model1 = Encoder(enc_in, emb_dim, hid_dim, n_layers, 0.2)
    # hidden, cell = model1(x)
    model2 = Decoder(dec_in, emb_dim, hid_dim, n_layers, 0.2)
    # model2(x, hidden, cell)
    y = torch.randn(batch_size, 10, dec_in)
    model = Lstm(model1, model2, torch.device("cuda"))
    output = model(x, y)