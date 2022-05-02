from layers.Embed import RotaryEmbedding
import torch.nn as nn
from utils.masking import TriangularCausalMask
import torch
from math import sqrt
import numpy as np
from utils.activation import Swish, Relu, Gelu
from utils.tools import ScaleOffset, attention_normalize
from args import args
import torch.nn.functional as F

class GateAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(GateAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.activation = args.test_activation
        
    def forward(self, u, q, k, v, attn_mask):
        B, L, E = q.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum('bmd,bnd->bmn', q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, gau=True, device=q.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(attention_normalize(scale * scores, dim=-1, method=self.activation))
        V = u * torch.einsum('bmn,bnd->bmd', A, v)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
            
class GateAttentionLayer(nn.Module):
    def __init__(self,
        attention,
        d_model,
        uv_size,
        qk_size,
        activation="gelu",
        use_bias=True,
        use_conv=True,
        use_aff = True,
        **kwargs):
        super(GateAttentionLayer, self).__init__()

        self.inner_attention = attention
        self.d_model = d_model
        self.uv_size = uv_size
        self.qk_size = qk_size
        self.use_bias = use_bias
        self.use_conv = use_conv
        self.use_aff = use_aff
        if activation == "relu":
            self.activation = Relu()
        elif activation == "gelu":
            self.activation = Gelu()
        else:
            self.activation = Swish()

        self.query_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.key_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.qk_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), self.activation)
        self.value_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), self.activation)

        self.rotary_emb = RotaryEmbedding(dim = self.qk_size, skip=False)

        if self.use_conv:
            self.o_dense = nn.Conv1d(in_channels=self.uv_size, out_channels=self.d_model, kernel_size=1)
            self.u_projection = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=self.uv_size,kernel_size=1), self.activation)
        else:
            self.o_dense = nn.Linear(self.uv_size, self.d_model, bias=self.use_bias)
            self.u_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), self.activation)

        self.q_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)

    def forward(self, u, queries, keys, values, attn_mask=None):
            
        # 投影变换
        if self.use_aff:
            qk = self.qk_projection(queries)
            q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
        else:
            q = self.query_projection(queries)
            k = self.key_projection(keys)
        
        # 加入RoPE
        q, k = self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)

        v = self.value_projection(values)
        if self.use_conv:
            u = self.u_projection(u.permute(0, 2, 1)).transpose(1, 2)
        else:
            u = self.u_projection(u)

        # Attention
        out, attn = self.inner_attention(u, q, k, v, attn_mask)
        # 计算输出
        if self.use_conv:
            o = self.o_dense(out.permute(0, 2, 1)).transpose(1, 2)
        else:
            o = self.o_dense(out)
        return o, attn


class MultiHeadGateAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(MultiHeadGateAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.activation = args.test_activation
        
    def forward(self, u, q, k, v, attn_mask):
        B, L, H, E = q.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, gau=True, device=q.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(attention_normalize(scale * scores, dim=-1, method=self.activation))
        V = u * torch.einsum("bhls,bshd->blhd", A, v)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
            
class MultiHeadGateAttentionLayer(nn.Module):
    def __init__(self,
        attention,
        d_model,
        n_heads,
        uv_size,
        qk_size,
        activation="gelu",
        use_bias=True,
        use_conv=True,
        use_aff = True,
        **kwargs):
        super(MultiHeadGateAttentionLayer, self).__init__()

        self.inner_attention = attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.uv_size = uv_size
        self.qk_size = qk_size
        self.use_bias = use_bias
        self.use_conv = use_conv
        self.use_aff = use_aff
        if activation == "relu":
            self.activation = Relu()
        elif activation == "gelu":
            self.activation = Gelu()
        else:
            self.activation = Swish()

        self.query_projection = nn.Sequential(nn.Linear(d_model, self.qk_size*self.n_heads, self.use_bias), self.activation)
        self.key_projection = nn.Sequential(nn.Linear(d_model, self.qk_size*self.n_heads, self.use_bias), self.activation)
        self.qk_projection = nn.Sequential(nn.Linear(d_model, self.qk_size*self.n_heads, self.use_bias), self.activation)
        self.value_projection = nn.Sequential(nn.Linear(d_model, self.uv_size*self.n_heads, self.use_bias), self.activation)

        self.rotary_emb = RotaryEmbedding(dim=self.qk_size*self.n_heads, skip=False)

        if self.use_conv:
            self.u_projection = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=self.uv_size*self.n_heads,kernel_size=1), self.activation)
            self.o_dense = nn.Conv1d(in_channels=self.uv_size*self.n_heads, out_channels=self.d_model, kernel_size=1)
        else:
            self.u_projection = nn.Sequential(nn.Linear(d_model, self.uv_size*self.n_heads, self.use_bias), self.activation)
            self.o_dense = nn.Linear(self.uv_size*self.n_heads, self.d_model, bias=self.use_bias)
            

        self.q_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)

    def forward(self, u, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 投影变换
        if self.use_aff:
            qk = self.qk_projection(queries)
            q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
        else:
            q = self.query_projection(queries)
            k = self.key_projection(keys)
        
        # 加入RoPE
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=1).view(B, L, H, -1)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=1).view(B, S, H, -1)

        v = self.value_projection(values).view(B, S, H, -1)
        if self.use_conv:
            u = self.u_projection(u.permute(0, 2, 1)).transpose(1, 2).view(B, L, H, -1)
        else:
            u = self.u_projection(u).view(B, L, H, -1)

        # Attention
        out, attn = self.inner_attention(u, q, k, v, attn_mask)
        out = out.view(B, L, -1)
        # 计算输出
        if self.use_conv:
            o = self.o_dense(out.permute(0, 2, 1)).transpose(1, 2)
        else:
            o = self.o_dense(out)
        return o, attn