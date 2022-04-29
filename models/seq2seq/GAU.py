from layers.Embed import DataEmbedding_wo_pos
from layers.GateAttention import GatedAttentionLayer
import torch.nn as nn
from layers.GAU_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.GateAttention import GatedAttentionLayer, GateAttention
import torch

class GAU_alpha(nn.Module):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    GatedAttentionLayer(
                        GateAttention(False, attention_dropout=args.dropout, output_attention=args.output_attention), 
                        args.d_model, args.uv_size, args.qk_size
                    ),
                    args.d_model,
                    dropout=args.dropout
                ) for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    GatedAttentionLayer(
                        GateAttention(True, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.uv_size, args.qk_size
                        ),
                    GatedAttentionLayer(
                        GateAttention(False, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.uv_size, args.qk_size
                        ),
                    args.d_model,
                    dropout=args.dropout
                )
                for l in range(args.d_layers)
            ],
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

    
    