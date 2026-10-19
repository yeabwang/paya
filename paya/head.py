"""Metadata-free bidirectional self-attention and signal cross-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def structural_encoding(positions, width):
    """Fixed sinusoidal encoding of prepared time/scale/CLS coordinates."""
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        math.ceil(width/6), device=positions.device, dtype=positions.dtype) / max(1, math.ceil(width/6)-1))
    angles = positions[..., None] * frequencies * (2*math.pi)
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)[..., :width]


class CrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query = nn.Linear(width, width)
        self.key, self.value = nn.Linear(768, width), nn.Linear(768, width)
        self.output = nn.Linear(width, width)

    def forward(self, text, signal, position_encoding, signal_mask):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        q = split(self.query(text))
        k = split(self.key(signal) + position_encoding)
        v = split(self.value(signal))
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=signal_mask[:, None, None, :],
                         dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.output(result.transpose(1, 2).reshape(text.shape))


class FusionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = CrossAttention(width, heads, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(),
                              nn.Dropout(dropout), nn.Linear(4*width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_mask, signal, signal_mask, positions):
        normalized = self.norms[0](text)
        attended = self.self_attention(normalized, normalized, normalized,
                                      key_padding_mask=~text_mask, need_weights=False)[0]
        text = text + self.dropout(attended)
        text = text + self.dropout(self.cross_attention(self.norms[1](text), signal, positions, signal_mask))
        text = text + self.dropout(self.feed_forward(self.norms[2](text)))
        return text.masked_fill(~text_mask[..., None], 0)


class DecisionHead(nn.Module):
    """Fuse text self-attention with signal cross-attention and score each choice."""

    def __init__(self, width=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or layers <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid decision-head dimensions or dropout")
        self.width = width
        self.config = dict(width=width, layers=layers, heads=heads, dropout=dropout)
        self.text_input = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.roles = nn.Parameter(torch.zeros(3, width))
        self.blocks = nn.ModuleList(FusionBlock(width, heads, dropout) for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.scorer = nn.Linear(width, 1)

    def forward(self, signal, signal_mask, positions, question, criteria, ending, criteria_mask):
        batch, count, hidden = signal.shape
        if hidden != 768 or signal_mask.shape != (batch, count) or positions.shape != (batch, count, 3):
            raise ValueError("Head expects signal [B,M,768], positions [B,M,3], and mask [B,M]")
        if question.shape != (batch, 768) or ending.shape != (batch, 768):
            raise ValueError("Question/ending must have shape [B,768]")
        if criteria.ndim != 3 or criteria.shape[0] != batch or criteria.shape[-1] != 768:
            raise ValueError("Criteria must have shape [B,K,768]")
        if criteria_mask.shape != criteria.shape[:2] or criteria_mask.dtype != torch.bool or signal_mask.dtype != torch.bool:
            raise ValueError("Boolean masks must match signal and criteria")
        if not signal_mask.any(-1).all() or not (criteria_mask.sum(-1) >= 2).all():
            raise ValueError("Need real signal vectors and at least two real criteria")
        if not all(torch.isfinite(x).all() for x in (signal, positions, question, criteria, ending)):
            raise ValueError("Head inputs must be finite")
        text = self.text_input(torch.cat((question[:, None], criteria, ending[:, None]), dim=1))
        roles = torch.cat((self.roles[:1], self.roles[1:2].expand(criteria.shape[1], -1), self.roles[2:]))
        text = text + roles[None]
        boundary_mask = torch.ones(batch, 1, dtype=torch.bool, device=signal.device)
        text_mask = torch.cat((boundary_mask, criteria_mask, boundary_mask), dim=1)
        position_encoding = structural_encoding(positions, self.width)
        for block in self.blocks:
            text = block(text, text_mask, signal, signal_mask, position_encoding)
        logits = self.scorer(self.final_norm(text[:, 1:-1])).squeeze(-1)
        if not torch.isfinite(logits[criteria_mask]).all():
            raise FloatingPointError("Nonfinite decision logits")
        return logits.masked_fill(~criteria_mask, -torch.inf)
