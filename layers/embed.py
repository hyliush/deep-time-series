import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from math import pi, log
from einops import rearrange, repeat
import math

class SpatialEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(SpatialEmbedding, self).__init__()
        self.spa_emb = nn.Embedding(c_in, d_model)
        
    def forward(self, x):
        x = x.long()
        spa_emb = self.spa_emb(x)
        
        return spa_emb

class PositionalEmbedding(nn.Module):
    '''local position'''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    '''
    use embedding as weight to transform timestamp features
    '''
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        '''
        month, monthday, weekday, hour, minute
        '''
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        self.freq = freq
        
        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if self.freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3, "10min":3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)
        
# helper functions
def exists(val):
    return val is not None
# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = torch.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        skip = False,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()
        self.skip = skip
        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def rotate_queries_or_keys(self, t, seq_dim=1):
        if self.skip:
            return t
        device = t.device
        seq_len = t.shape[seq_dim]
        # SinusoidalPositionEmbedding
        freqs = self.forward(lambda: torch.arange(seq_len, device = device), cache_key = seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs