import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, args):
        super(Autoformer, self).__init__()
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Decomp
        kernel_size = args.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq,
                                                  args.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq,
                                                  args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                        output_attention=args.output_attention),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    moving_avg=args.moving_avg,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=my_Layernorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, args.factor, attention_dropout=args.dropout,
                                        output_attention=False),
                        args.d_model, args.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                        output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.out_size,
                    args.d_ff,
                    moving_avg=args.moving_avg,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=my_Layernorm(args.d_model),
            projection=nn.Linear(args.d_model, args.out_size, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
