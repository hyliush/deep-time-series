import torch
from args import args

def make_testdata():
    x_enc = torch.randn(args.batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(args.batch_size, args.seq_len, 3)
    x_dec = torch.randn(args.batch_size, args.label_len+args.pred_len, args.dec_in)
    x_mark_dec = torch.randn(args.batch_size, args.label_len+args.pred_len, 3)
    return x_enc, x_mark_enc, x_dec, x_mark_dec