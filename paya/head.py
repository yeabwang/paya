"""Question/candidate self-attention followed by signal cross-attention."""

import torch
from torch import nn


class AttentionBlock(nn.Module):
    """Pre-norm residual attention and a residual feed-forward layer."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, query, memory, memory_mask):
        normalized = self.norm(query)
        keys = normalized if memory is None else memory
        values, _ = self.attention(
            normalized, keys, keys, key_padding_mask=~memory_mask, need_weights=False
        )
        query = query + self.drop(values)
        return query + self.drop(self.ff(query))


class AttentionHead(nn.Module):
    """Contextualize text, retrieve signal evidence, and score each candidate."""

    def __init__(self, width=128, dropout=0.1, heads=4):
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 2
            or isinstance(heads, bool)
            or not isinstance(heads, int)
            or heads < 1
            or width % heads
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid attention width, heads or dropout")
        self.config = dict(
            kind="self_then_cross_attention", width=width, dropout=dropout, heads=heads
        )
        self.signal = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width))
        self.text = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.roles = nn.Parameter(torch.zeros(2, width))
        self.cross_attention = AttentionBlock(width, heads, dropout)
        self.self_attention = AttentionBlock(width, heads, dropout)
        self.final_norm = nn.LayerNorm(width)
        self.score = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, signal, signal_mask, question, criteria, criteria_mask):
        batch = len(question)
        if (
            signal.ndim != 3
            or signal.shape[0] != batch
            or signal.shape[-1] != 512
            or signal_mask.shape != signal.shape[:2]
            or signal_mask.dtype != torch.bool
            or not signal_mask.any(-1).all()
        ):
            raise ValueError(
                "Need signal tokens [B,T,512] with a nonempty boolean mask"
            )
        if (
            question.shape != (batch, 768)
            or criteria.ndim != 3
            or criteria.shape[0] != batch
            or criteria.shape[-1] != 768
            or criteria_mask.shape != criteria.shape[:2]
            or criteria_mask.dtype != torch.bool
            or not (criteria_mask.sum(-1) >= 2).all()
        ):
            raise ValueError("Need separate 768-D text vectors and two real candidates")
        if not all(
            torch.isfinite(value).all() for value in (signal, question, criteria)
        ):
            raise ValueError("Head inputs must be finite")
        memory = self.signal(signal)
        queries = torch.cat(
            [
                (self.text(question) + self.roles[0])[:, None],
                self.text(criteria) + self.roles[1],
            ],
            1,
        )
        valid = torch.cat([torch.ones_like(criteria_mask[:, :1]), criteria_mask], 1)
        queries = self.self_attention(queries, None, valid)
        queries = self.cross_attention(queries, memory, signal_mask)
        queries = self.final_norm(queries)
        q, c = queries[:, :1], queries[:, 1:]
        logits = self.score(torch.cat([q.expand_as(c), c, q * c], -1)).squeeze(-1)
        if not torch.isfinite(logits[criteria_mask]).all():
            raise FloatingPointError("Nonfinite decision logits")
        return logits.masked_fill(~criteria_mask, -torch.inf)
