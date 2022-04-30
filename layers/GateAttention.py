from layers.Embed import RotaryEmbedding
import torch.nn as nn
from utils.masking import TriangularCausalMask
import torch
from math import sqrt
import numpy as np
from utils.activation import Swish
from utils.tools import ScaleOffset, attention_normalize

class GateAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(GateAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, u, q, k, v, attn_mask):
        B, L, E = q.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum('bmd,bnd->bmn', q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, gau=True, device=q.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(attention_normalize(scale * scores, axis=-1))
        V = u * torch.einsum('bmn,bnd->bmd', A, v)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
            
class GatedAttentionLayer(nn.Module):
    def __init__(self,
        attention,
        d_model,
        uv_size,
        qk_size,
        use_bias=True,
        use_conv=True,
        **kwargs):
        super(GatedAttentionLayer, self).__init__()

        self.inner_attention = attention
        self.d_model = d_model
        self.uv_size = uv_size
        self.qk_size = qk_size
        self.use_bias = use_bias
        self.use_conv = use_conv

        self.query_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), Swish())
        self.key_projection = nn.Sequential(nn.Linear(d_model, self.qk_size, self.use_bias), Swish())
        self.value_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), Swish())
        self.rotary_emb = RotaryEmbedding(dim = self.qk_size)
        
        if self.use_conv:
            self.o_dense = nn.Conv1d(in_channels=self.uv_size, out_channels=self.d_model, kernel_size=1)
            self.u_projection = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=self.uv_size,kernel_size=1), Swish())
        else:
            self.o_dense = nn.Linear(self.uv_size, self.d_model, bias=self.use_bias)
            self.u_projection = nn.Sequential(nn.Linear(d_model, self.uv_size, self.use_bias), Swish())

        self.q_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(self.qk_size, offset=self.use_bias)

    def forward(self, u, queries, keys, values, attn_mask=None):
            
        # 投影变换
        q = self.query_projection(queries)
        k = self.key_projection(keys)
        v = self.value_projection(values)
        if self.use_conv:
            u = self.u_projection(u.permute(0, 2, 1)).transpose(1, 2)
        else:
            u = self.u_projection(u)
        # q, k = self.q_scaleoffset(q), self.k_scaleoffset(k)
        # 加入RoPE
        q, k = self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)
        # Attention
        out, attn = self.inner_attention(u, q, k, v, attn_mask)
        # 计算输出
        if self.use_conv:
            o = self.o_dense(out.permute(0, 2, 1)).transpose(1, 2)
        else:
            o = self.o_dense(out)
        return o, attn