"""One frozen SPOTR global token per native multichannel recording."""

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
    values: torch.Tensor             # [1,512], one global token per recording
    intervals_s: torch.Tensor        # [1,2], elapsed start/end seconds before internal padding


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
        samples, intervals = encoder_input.samples, encoder_input.intervals_s
        if (intervals.shape != (1, 2) or not torch.isfinite(intervals).all()
                or (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise ValueError("SPOTR needs one valid elapsed start/end interval")
        if (samples.ndim != 2 or not 1 <= samples.shape[0] <= SPOTR_MAX_CHANNELS
                or not 2 <= samples.shape[1] <= SPOTR_MAX_SAMPLES or not torch.isfinite(samples).all()):
            raise ValueError("SPOTR expects finite [C,L] samples: 1–128 channels, 2–12800 samples")
        channels, length = samples.shape
        tokens = self.core(samples.unsqueeze(0))
        patches = 2*((length+199)//200)
        if tokens.shape != (1, 1+channels+patches, self.width) or not torch.isfinite(tokens).all():
            raise ValueError("Pinned SPOTR must return finite tokens [1,1+C+N,512]")
        return EncodedSignal(tokens[:, 0], intervals)
