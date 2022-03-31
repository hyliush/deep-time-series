import torch.nn as nn
import random
import torch
import torch.nn.functional as F
import sys
from layers.Embed import DataEmbedding_ED

class Encoder(nn.Module):
    def __init__(self, enc_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout):
        super().__init__()
        
        self.embedding = DataEmbedding_ED(enc_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
    def forward(self, x_enc, x_enc_mark):
        
        #x_enc = [x_enc len, batch size]
        
        embedded = self.embedding(x_enc, x_enc_mark)
        #embedded = [x_enc len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [x_enc len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [x_enc len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, x_enc len, enc hid dim * 2]
        
        batch_size, x_enc_len = encoder_outputs.shape[0], encoder_outputs.shape[1]
        
        #repeat decoder hidden state x_enc_len times
        hidden = hidden.unsqueeze(1).repeat(1, x_enc_len, 1)
        
        #hidden = [batch size, x_enc len, dec hid dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, x_enc len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, x_enc len]
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, dec_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout, attention):
        super().__init__()

        self.attention = attention
        
        self.embedding = DataEmbedding_ED(dec_in, emb_dim, embed, freq, dropout)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, dec_in)
        
    def forward(self, input, input_mark, hidden, encoder_outputs):
             
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, x_enc len, enc hid dim * 2]
        
        #input = [1, batch size]
        
        embedded = self.embedding(input, input_mark)
        
        #embedded = [batch size, 1, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, x_enc len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, x_enc len]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [batch size, 1, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden.transpose(1, 0)).all()
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 2))
        
        #prediction = [batch size, seq_len, output dim]
        return prediction, hidden.squeeze(0)

class GruAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        enc_in, dec_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout = \
        args.enc_in, args.dec_in, args.d_model, args.d_model, args.d_model, args.embed, \
        args.freq, args.dropout
        self.pred_len = args.pred_len
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        
        attention = Attention(enc_hid_dim, dec_hid_dim)
        self.encoder = Encoder(enc_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout)
        self.decoder = Decoder(dec_in, emb_dim, enc_hid_dim, dec_hid_dim, embed, freq, dropout, attention)
        
    def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark):
        
        #x_enc = [x_enc len, batch size, n_features]
        #x_dec = [x_dec len, batch size, n_features]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        if self.training:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0
        batch_size, x_dec_len, dec_in = x_dec.shape
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, x_dec_len-1, dec_in).to(x_enc.device)
        
        #last hidden state of the encoder is the context
        encoder_outputs, hidden = self.encoder(x_enc, x_enc_mark)
        
        input = x_dec[:, 0, :].unsqueeze(dim=1)
        input_mark = x_dec_mark[:, 0, :].unsqueeze(dim=1)
        for t in range(1, x_dec_len):
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, input_mark, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t-1, :] = output.squeeze(dim=1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
            input_mark = x_dec_mark[:, t, :].unsqueeze(dim=1)

        return outputs[:, -self.pred_len:, :]
       
if __name__ == '__main__':
    enc_in, dec_in, emb_dim, enc_hid_dim, dec_hid_dim = 45, 45, 512, 512, 512
    batch_size, seq_len = 32, 10
    x = torch.randn(batch_size, seq_len, enc_in)
    model1 = Encoder(enc_in, emb_dim, enc_hid_dim, dec_hid_dim, 0.2)
    # hidden, cell = model1(x)
    atten = Attention(enc_hid_dim, dec_hid_dim)
    model2 = Decoder(dec_in, emb_dim, enc_hid_dim, dec_hid_dim, 0.2,atten)
    # model2(x, hidden, cell)
    y = torch.randn(batch_size, 10, dec_in)
    model = GruAttention(model1, model2, torch.device("cuda"))
    output = model(x, y)
    print("")