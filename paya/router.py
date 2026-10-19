"""Two-expert top-1 gating from processed waveforms and a question."""

import torch
from torch import nn


class ExpertRouter(nn.Module):
    """Score two eligible experts using waveform convolutions and the question."""

    def __init__(self):
        super().__init__()
        self.waveform = nn.Sequential(nn.Conv1d(1, 16, 9, 2, 4), nn.GELU(),
                                      nn.Conv1d(16, 32, 5, 2, 2), nn.GELU())
        self.question = nn.Sequential(nn.Linear(768, 128), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, frames, owners, question, compatible):
        batch = question.shape[0]
        if question.shape != (batch, 768) or compatible.shape != (batch, 2):
            raise ValueError("Router expects questions [B,768] and compatibility [B,2]")
        if compatible.dtype != torch.bool or not compatible.any(-1).all():
            raise ValueError("Every request needs at least one compatible expert")
        if not torch.isfinite(question).all():
            raise ValueError("Nonfinite question embeddings")
        features = []
        for index, length in enumerate((390, 512)):
            x, owner = frames[index], owners[index]
            if x.shape != (owner.numel(), 1, length) or owner.dtype != torch.long:
                raise ValueError("Invalid real-frame router input")
            if owner.numel() and (owner.min() < 0 or owner.max() >= batch):
                raise ValueError("Router frame owner is out of range")
            count = torch.bincount(owner, minlength=batch)
            if not torch.equal(count > 0, compatible[:, index]):
                raise ValueError("Compatibility must match the available frame owners")
            pooled = question.new_zeros(batch, 32)
            if x.numel():
                if not torch.isfinite(x).all():
                    raise ValueError("Router waveforms must be finite")
                pooled = pooled.index_add(0, owner, self.waveform(x).mean(-1))
                pooled = pooled / count.clamp_min(1)[:, None]
            features.append(pooled)
        logits = self.gate(torch.cat([*features, self.question(question)], dim=-1))
        if not torch.isfinite(logits[compatible]).all():
            raise FloatingPointError("Nonfinite router logits")
        return logits.masked_fill(~compatible, -torch.inf)
