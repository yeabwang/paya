"""SPOTR encoder from the pinned MIT-licensed upstream implementation."""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class LayerNorm2D(nn.Module):

    def __init__(self, num_channels: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).to(torch.float32).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var.to(dtype) + self.eps)
        return x_hat * self.weight[:, None, None] + self.bias[:, None, None]

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(4 * dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Attention(nn.Module):

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(self, x_q, x_k, x_v):
        batch_size, _, _ = x_q.shape
        xq, xk, xv = (self.wq(x_q), self.wk(x_k), self.wv(x_v))
        xq = xq.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xk = xk.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xv = xv.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(xq, xk, xv).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return self.wo(output)

class TransformerEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads=8):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = Attention(dim=dim, n_heads=num_heads)
        self.mlp = MLP(dim=dim)
        self.input_norm = RMSNorm(dim)
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x):
        res = x
        x = self.input_norm(x)
        x = res + self.self_attn(x, x, x)
        out = x + self.mlp(self.post_attention_layernorm(x))
        return out

class SPOTREncoder(nn.Module):

    def __init__(self, encoder_dim=256, patch_size=100, num_heads=8, n_layers=12):
        super(SPOTREncoder, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 25), padding=(0, 12), bias=False), LayerNorm2D(num_channels=32), nn.GELU(approximate='tanh'), nn.Conv2d(in_channels=32, out_channels=encoder_dim, kernel_size=(1, patch_size), stride=(1, patch_size), bias=False), LayerNorm2D(num_channels=encoder_dim), nn.GELU(approximate='tanh'))
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, 128, encoder_dim), requires_grad=True)
        self.spatial_attn = Attention(dim=encoder_dim, n_heads=8)
        self.temporal_attn = Attention(dim=encoder_dim, n_heads=8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.block = nn.ModuleList([TransformerEncoderBlock(dim=encoder_dim, num_heads=num_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(encoder_dim)

    def encode(self, x):
        b, c, l = x.shape
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x = (x - x_mean) / (x_std + 1e-06)
        remainder = 200 * math.ceil(l / 200) - l
        if remainder > 0:
            x = F.pad(x, (0, remainder), mode='constant', value=0)
        x_embedding = self.conv(x.unsqueeze(dim=1))
        x_embedding = rearrange(x_embedding, 'b d c n -> b c n d')
        b, c, n, d = x_embedding.shape
        x_embedding = x_embedding + self.pos_embed[:, :c, :n, :]
        x_spatial_embedding = rearrange(x_embedding, 'b c n d -> (b c) n d')
        x_temporal_embedding = rearrange(x_embedding, 'b c n d -> (b n) c d')
        x_spatial_embedding = x_spatial_embedding + self.spatial_attn(x_spatial_embedding, x_spatial_embedding, x_spatial_embedding)
        x_spatial_embedding = torch.mean(x_spatial_embedding, dim=1)
        x_temporal_embedding = x_temporal_embedding + self.temporal_attn(x_temporal_embedding, x_temporal_embedding, x_temporal_embedding)
        x_temporal_embedding = torch.mean(x_temporal_embedding, dim=1)
        x_spatial_embedding = rearrange(x_spatial_embedding, '(b c) d -> b c d', b=b, c=c, d=d)
        x_temporal_embedding = rearrange(x_temporal_embedding, '(b n) d -> b n d', b=b, n=n, d=d)
        x_embedding = torch.cat([self.cls_token.expand(b, -1, -1), x_spatial_embedding, x_temporal_embedding], dim=1)
        for block in self.block:
            x_embedding = block(x_embedding)
        return (x_embedding, x)

    def forward(self, x):
        x_embedding, x = self.encode(x)
        return x_embedding
