"""Choose one encoder from native cleaned waveforms and the question."""

import torch
from torch import nn


class ExpertRouter(nn.Module):
    """Pool native waveform convolutions and score two eligible encoders."""

    def __init__(self):
        super().__init__()
        self.waveform = nn.Sequential(nn.Conv1d(1, 16, 9, 2, 4), nn.GELU(),
                                      nn.Conv1d(16, 32, 5, 2, 2), nn.GELU())
        self.question = nn.Sequential(nn.Linear(768, 128), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, signals, owners, question, compatible):
        batch = question.shape[0]
        if question.shape != (batch, 768) or compatible.shape != (batch, 2):
            raise ValueError("Router expects questions [B,768] and compatibility [B,2]")
        if compatible.dtype != torch.bool or not compatible.any(-1).all():
            raise ValueError("Every request needs at least one compatible encoder")
        if not torch.isfinite(question).all():
            raise ValueError("Nonfinite question embeddings")
        if not isinstance(signals, (list, tuple)) or len(signals) != owners.numel() or owners.dtype != torch.long:
            raise ValueError("Supply native signal arrays and matching integer channel owners")
        if not owners.numel() or owners.min() < 0 or owners.max() >= batch:
            raise ValueError("Router channel owner is out of range")
        counts = torch.bincount(owners, minlength=batch)
        if not (counts > 0).all():
            raise ValueError("Every request needs a cleaned signal channel")

        groups = {}
        for index, signal in enumerate(signals):
            if signal.ndim != 1 or len(signal) < 2 or not torch.isfinite(signal).all():
                raise ValueError("Each native signal must be a finite one-dimensional array")
            groups.setdefault(len(signal), []).append(index)
        pooled = question.new_zeros(batch, 32)
        # Batch equal-length channels without padding or encoder-specific framing.
        for indices in groups.values():
            values = torch.stack([signals[i] for i in indices])[:, None]
            convolved = self.waveform(values).mean(-1)
            pooled = pooled.index_add(0, owners[indices], convolved)
        pooled = pooled / counts[:, None]
        logits = self.gate(torch.cat((pooled, self.question(question)), dim=-1))
        if not torch.isfinite(logits[compatible]).all():
            raise FloatingPointError("Nonfinite router logits")
        return logits.masked_fill(~compatible, -torch.inf)
