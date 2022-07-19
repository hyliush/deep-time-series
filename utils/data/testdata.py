import torch
freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3, "10min":3}
def make_costumizeddata(batch_size, seq_len, label_len, pred_len, enc_in, dec_in, fd):
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, fd)
    x_dec = torch.randn(batch_size, label_len+pred_len, dec_in)
    x_mark_dec = torch.randn(batch_size, label_len+pred_len, fd)
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def make_testdata(args):
    if not hasattr(args, "freq"):
        args.freq = "b"
    fd = freq_map[args.freq]
    x_enc = torch.randn(args.batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(args.batch_size, args.seq_len, fd)
    x_dec = torch.randn(args.batch_size, args.label_len+args.pred_len, args.dec_in)
    x_mark_dec = torch.randn(args.batch_size, args.label_len+args.pred_len, fd)
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def make_allone(args):
    fd = freq_map[args.freq]
    x_enc = torch.ones(args.batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.ones(args.batch_size, args.seq_len, fd)
    x_dec = torch.ones(args.batch_size, args.label_len+args.pred_len, args.dec_in)
    x_mark_dec = torch.ones(args.batch_size, args.label_len+args.pred_len, fd)
    return x_enc, x_mark_enc, x_dec, x_mark_dec
