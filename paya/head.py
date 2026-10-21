"""MLP fusion of frozen signal, question and candidate embeddings."""

import torch
from torch import nn


class MLPFusionHead(nn.Module):
    """Score candidates with an MLP and Hadamard product features."""

    def __init__(self, width=128, dropout=0.1):
        super().__init__()
        if width < 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid MLP dimensions or dropout")
        self.config = dict(kind="interaction", width=width, dropout=dropout)
        self.signal = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width), nn.GELU())
        self.text = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.context = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(dropout))
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, signal, signal_mask, question, criteria, criteria_mask):
        batch = len(signal)
        if signal.shape != (batch, 1, 512) or signal_mask.shape != (batch, 1) or not signal_mask.all():
            raise ValueError("MLP head requires one real signal vector")
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
