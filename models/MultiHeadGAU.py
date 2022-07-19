from layers.Embed import DataEmbedding_wo_pos, DataEmbedding
import torch.nn as nn
from layers.GateAttention_Family import MultiHeadGateAttentionLayer, MultiHeadGateAttention
import torch
from layers.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayer
from layers.GAU_EncDec import DecoderLayer as DecoderLayer1
from layers.Transformer_EncDec import DecoderLayer as DecoderLayer2

class MultiHeadGAU(nn.Module):
    def __init__(self, args):
        from layers.GAU_EncDec import Decoder, Encoder, EncoderLayer, ConvLayer
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadGateAttentionLayer(
                        MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), 
                        args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation,
                        args.use_bias, args.use_conv, args.use_aff
                    ),
                    args.d_model,
                    dropout=args.dropout
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)

    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        out = self.projection(enc_out[:, -1:, :])
        return out


class AU(nn.Module):
    def __init__(self, args):
        super(AU, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        from layers.Transformer_EncDec import Decoder, Encoder, EncoderLayer, ConvLayer

        # Embedding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), 
                        args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.projection = nn.Linear(args.d_model, args.out_size, bias=True)

    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        out = self.projection(enc_out[:, -1:, :])
        return out

