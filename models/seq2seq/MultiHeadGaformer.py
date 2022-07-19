from layers.Embed import DataEmbedding_wo_pos, DataEmbedding
import torch.nn as nn
from layers.GAU_EncDec import Decoder, Encoder, EncoderLayer, ConvLayer
from layers.GateAttention_Family import MultiHeadGateAttentionLayer, MultiHeadGateAttention
import torch
from layers.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayer
from layers.GAU_EncDec import DecoderLayer as DecoderLayer1
from layers.Transformer_EncDec import DecoderLayer as DecoderLayer2

class MultiHeadGaformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadGateAttentionLayer(
                        MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), 
                        args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation,
                        args.use_bias, args.use_conv, args.use_aff,
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
        
        # Decoder
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        if args.dec_selfattn == 'gate':
            selfattnlayer = MultiHeadGateAttentionLayer(
                            MultiHeadGateAttention(True, attention_dropout=args.dropout, output_attention=False),
                            args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation,
                            args.use_bias, args.use_conv, args.use_aff)
            self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq,
                                        args.dropout)
        else:
            Attn = ProbAttention if args.dec_selfattn == "prob" else FullAttention
            selfattnlayer = AttentionLayer(
                Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                args.d_model, args.n_heads, mix=args.mix)

        if args.dec_crossattn == "gate":
            "gateAttenLayer 包含FNN, 因此用GAU Decoderlayer 剔除FNN"
            crossattnlayer = MultiHeadGateAttentionLayer(
                            MultiHeadGateAttention(False, attention_dropout=args.dropout, output_attention=False),
                            args.d_model, args.n_heads, args.uv_size, args.qk_size, args.activation,
                            args.use_bias, args.use_conv, args.use_aff)
            decoderlayer = DecoderLayer1( selfattnlayer, crossattnlayer, args.d_model, dropout=args.dropout)
        else:
            "不包含FNN，因此用Transformer Decoderlayer"
            Attn = ProbAttention if args.dec_crossattn == "prob" else FullAttention
            crossattnlayer = AttentionLayer(
                Attn(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                args.d_model, args.n_heads, mix=False)
            
            decoderlayer = DecoderLayer2(selfattnlayer,crossattnlayer, args.d_model, args.d_ff, dropout=args.dropout,
                            activation=args.activation)
                        
        self.decoder = Decoder(
            [decoderlayer for l in range(args.d_layers)],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.out_size, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    
    