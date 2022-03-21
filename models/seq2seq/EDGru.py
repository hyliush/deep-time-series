import torch.nn as nn
import random
import torch
import sys
sys.path.append("d:\\IDEA\\Spatial-temporal\\deep-time-series")
from layers.Embed import DataEmbedding_ED

class Encoder(nn.Module):
    def __init__(self, enc_in, emb_dim, hid_dim, embed, freq, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = DataEmbedding_ED(enc_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        
    def forward(self, x_enc, x_mark_enc):
        
        #x_enc = [x_enc len, batch size]
        
        embedded = self.embedding(x_enc, x_mark_enc)
        #embedded = [x_enc len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [x_enc len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden
class Decoder(nn.Module):
    def __init__(self, dec_in, emb_dim, hid_dim, embed, freq, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = DataEmbedding_ED(dec_in, emb_dim, embed, freq, dropout)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, dec_in)
        
    def forward(self, input, input_mark, hidden, context):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        embedded = self.embedding(input, input_mark)
        #embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context.transpose(0, 1)), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        output = torch.cat((embedded, hidden.transpose(0, 1), context.transpose(0, 1)), 
                           dim = 2)
        
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Gru(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        enc_in, dec_in, emb_dim, hid_dim, embed, freq, dropout = \
        args.enc_in, args.dec_in, args.d_model, args.d_model, \
        args.embed, args.freq, args.dropout
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.out_size = args.out_size
        self.encoder = Encoder(enc_in, emb_dim, hid_dim, embed, freq, dropout)
        self.decoder = Decoder(dec_in, emb_dim, hid_dim, embed, freq, dropout)
        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
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
        context = self.encoder(x_enc, x_mark_enc)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input, input_mark = x_dec[:, 0, :].unsqueeze(dim=1), x_mark_dec[:, 0, :].unsqueeze(dim=1)
        
        for t in range(1, x_dec_len):
            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, input_mark, hidden, context)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t-1, :] = output.squeeze(dim=1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = x_dec[:, t, :].unsqueeze(dim=1) if teacher_force else output
            input_mark = x_mark_dec[:, t, :].unsqueeze(dim=1)
        return outputs[:, :, -self.out_size:]
if __name__ == '__main__':

    enc_in, dec_in, emb_dim, hid_dim, n_layers = 45, 45, 512, 64, 2
    batch_size, seq_len = 32, 10
    x = torch.randn(batch_size, seq_len, enc_in)
    model1 = Encoder(enc_in, emb_dim, hid_dim, 0.2)
    # hidden, cell = model1(x)
    model2 = Decoder(dec_in, emb_dim, hid_dim, 0.2)
    # model2(x, hidden, cell)
    y = torch.randn(batch_size, 10, dec_in)
    model = Gru(model1, model2, torch.device("cuda"))
    output = model(x, y)
    print("")