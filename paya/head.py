"""Text self-attention and signal cross-attention."""

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query = nn.Linear(width, width)
        self.key, self.value = nn.Linear(512, width), nn.Linear(512, width)
        self.output = nn.Linear(width, width)

    def forward(self, text, signal, signal_mask):
        if not self.training and signal.shape[1] == 1:
            # A one-key softmax is one; avoid unused query/key work at inference.
            value = self.value(signal) * signal_mask[..., None]
            return self.output(value.expand(-1, text.shape[1], -1))
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        q = split(self.query(text))
        k = split(self.key(signal))
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

    def forward(self, text, text_mask, signal, signal_mask):
        normalized = self.norms[0](text)
        attended = self.self_attention(normalized, normalized, normalized,
                                      key_padding_mask=~text_mask, need_weights=False)[0]
        text = text + self.dropout(attended)
        text = text + self.dropout(self.cross_attention(self.norms[1](text), signal, signal_mask))
        text = text + self.dropout(self.feed_forward(self.norms[2](text)))
        return text.masked_fill(~text_mask[..., None], 0)


class DecisionHead(nn.Module):
    """Fuse text self-attention with signal cross-attention and score each choice."""

    def __init__(self, width=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        if width < 2 or heads <= 0 or width % heads or layers <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid decision-head dimensions or dropout")
        self.width = width
        self.config = dict(width=width, layers=layers, heads=heads, dropout=dropout)
        self.text_input = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.roles = nn.Parameter(torch.zeros(2, width))
        self.blocks = nn.ModuleList(FusionBlock(width, heads, dropout) for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.scorer = nn.Linear(width, 1)

    def forward(self, signal, signal_mask, question, criteria, criteria_mask):
        batch, count, hidden = signal.shape
        if hidden != 512 or signal_mask.shape != (batch, count):
            raise ValueError("Head expects signal [B,M,512] and mask [B,M]")
        if question.shape != (batch, 768):
            raise ValueError("Question must have shape [B,768]")
        if criteria.ndim != 3 or criteria.shape[0] != batch or criteria.shape[-1] != 768:
            raise ValueError("Criteria must have shape [B,K,768]")
        if criteria_mask.shape != criteria.shape[:2] or criteria_mask.dtype != torch.bool or signal_mask.dtype != torch.bool:
            raise ValueError("Boolean masks must match signal and criteria")
        if not signal_mask.any(-1).all() or not (criteria_mask.sum(-1) >= 2).all():
            raise ValueError("Need real signal vectors and at least two real criteria")
        if not all(torch.isfinite(x).all() for x in (signal, question, criteria)):
            raise ValueError("Head inputs must be finite")
        text = self.text_input(torch.cat((question[:, None], criteria), dim=1))
        roles = torch.cat((self.roles[:1], self.roles[1:].expand(criteria.shape[1], -1)))
        text = text + roles[None]
        question_mask = torch.ones(batch, 1, dtype=torch.bool, device=signal.device)
        text_mask = torch.cat((question_mask, criteria_mask), dim=1)
        for block in self.blocks:
            text = block(text, text_mask, signal, signal_mask)
        logits = self.scorer(self.final_norm(text[:, 1:])).squeeze(-1)
        if not torch.isfinite(logits[criteria_mask]).all():
            raise FloatingPointError("Nonfinite decision logits")
        return logits.masked_fill(~criteria_mask, -torch.inf)


class InteractionHead(nn.Module):
    """Score candidates through explicit interactions with the global signal."""

    def __init__(self, width=128, dropout=0.1):
        super().__init__()
        if width < 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid interaction-head dimensions or dropout")
        self.config = dict(kind="interaction", width=width, dropout=dropout)
        self.signal = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width), nn.GELU())
        self.text = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.context = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(dropout))
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, signal, signal_mask, question, criteria, criteria_mask):
        batch = len(signal)
        if signal.shape != (batch, 1, 512) or signal_mask.shape != (batch, 1) or not signal_mask.all():
            raise ValueError("Interaction head requires one unpadded global signal token")
        if question.shape != (batch, 768) or criteria.ndim != 3 or criteria.shape[0] != batch or criteria.shape[-1] != 768:
            raise ValueError("Question and criteria must have separate 768-dimensional vectors")
        if (criteria_mask.shape != criteria.shape[:2] or criteria_mask.dtype != torch.bool
                or signal_mask.dtype != torch.bool or not (criteria_mask.sum(-1) >= 2).all()):
            raise ValueError("Need boolean masks and at least two real criteria")
        if not all(torch.isfinite(x).all() for x in (signal, question, criteria)):
            raise ValueError("Head inputs must be finite")
        x, q, c = self.signal(signal[:, 0]), self.text(question), self.text(criteria)
        context = self.context(torch.cat([x, q, x * q], -1))[:, None].expand_as(c)
        z = self.score(torch.cat([context, c, context * c], -1)).squeeze(-1)
        if not torch.isfinite(z[criteria_mask]).all():
            raise FloatingPointError("Nonfinite decision logits")
        return z.masked_fill(~criteria_mask, -torch.inf)
