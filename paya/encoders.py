"""Frozen SPOTR global tokens from native multichannel recordings."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from .provenance import SPOTR_ID, SPOTR_REVISION, SPOTR_CONFIG, SPOTR_MAX_CHANNELS, SPOTR_MAX_SAMPLES, verify_source
from .vendor.spotr import SPOTREncoder as SPOTRCore


@dataclass
class EncodedSignal:
    values: torch.Tensor             # [M,512], one global token per channel group/window
    intervals_s: torch.Tensor        # [M,2], elapsed start/end seconds before internal padding


class SPOTREncoder(nn.Module):
    width = 512

    def __init__(self, core):
        super().__init__()
        self.core = core.requires_grad_(False).eval()

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        verify_source(directory, SPOTR_ID, SPOTR_REVISION)
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        if config != SPOTR_CONFIG:
            raise ValueError("SPOTR configuration must match the pinned Base checkpoint")
        core = SPOTRCore(**config)
        core.load_state_dict(load_file(str(directory/"model.safetensors")), strict=True)
        return cls(core)

    @torch.no_grad()
    def forward(self, encoder_input):
        frames, intervals = encoder_input.frames, encoder_input.intervals_s
        if not frames:
            raise ValueError("SPOTR needs at least one native window")
        if (intervals.shape != (len(frames), 2) or not torch.isfinite(intervals).all()
                or (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise ValueError("SPOTR windows need valid elapsed start/end intervals")
        groups = {}
        for index, frame in enumerate(frames):
            if (frame.ndim != 2 or not 1 <= frame.shape[0] <= SPOTR_MAX_CHANNELS
                    or not 2 <= frame.shape[1] <= SPOTR_MAX_SAMPLES or not torch.isfinite(frame).all()):
                raise ValueError("SPOTR expects finite [C,L] windows: 1–128 channels, 2–12800 samples")
            groups.setdefault(tuple(frame.shape), []).append(index)
        values = frames[0].new_empty(len(frames), self.width)
        for (channels, length), indices in groups.items():
            # Bound each batch by one full-capacity SPOTR input volume.
            batch_size = (SPOTR_MAX_CHANNELS*SPOTR_MAX_SAMPLES) // (channels*length)
            patches = 2*((length+199)//200)
            for start in range(0, len(indices), batch_size):
                current = indices[start:start+batch_size]
                tokens = self.core(torch.stack([frames[i] for i in current]))
                if tokens.shape != (len(current), 1+channels+patches, self.width) or not torch.isfinite(tokens).all():
                    raise ValueError("Pinned SPOTR must return finite tokens [G,1+C+N,512]")
                values[current] = tokens[:, 0]
        return EncodedSignal(values, intervals)
