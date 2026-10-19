"""Frozen MantisV2 embeddings from native windows with elapsed interval times."""

from dataclasses import dataclass

import torch
from torch import nn

from .provenance import MANTIS_ID, MANTIS_REVISION, verify_source


@dataclass
class EncodedSignal:
    values: torch.Tensor             # [M, 512]
    intervals_s: torch.Tensor        # [M,2], elapsed start/end seconds


class MantisEncoder(nn.Module):
    width = 512

    def __init__(self, core, max_batch_samples=1_000_000):
        super().__init__()
        self.core = core.requires_grad_(False).eval()
        if isinstance(max_batch_samples, bool) or not isinstance(max_batch_samples, int) or max_batch_samples < 64:
            raise ValueError("Encoder batch budget must allow at least 64 native samples")
        self.max_batch_samples = max_batch_samples

    @classmethod
    def load(cls, directory, max_batch_samples=1_000_000):
        from mantis.architecture import MantisV2
        verify_source(directory, MANTIS_ID, MANTIS_REVISION)
        core = MantisV2(device="cpu", return_transf_layer=2, output_token="combined", pre_training=False)
        core = core.from_pretrained(str(directory), local_files_only=True)
        core.remove_transf_layers()
        return cls(core, max_batch_samples)

    @torch.no_grad()
    def forward(self, encoder_input):
        frames = encoder_input.frames
        if not frames:
            raise ValueError("Mantis needs at least one native window")
        intervals = encoder_input.intervals_s
        if (intervals.shape != (len(frames), 2) or not torch.isfinite(intervals).all()
                or (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise ValueError("Mantis windows need valid elapsed start/end intervals")
        groups = {}
        for index, frame in enumerate(frames):
            if (frame.ndim != 2 or frame.shape[0] != 1 or frame.shape[1] % 32
                    or not 64 <= frame.shape[1] <= self.max_batch_samples or not torch.isfinite(frame).all()):
                raise ValueError("Mantis expects finite windows [1,L], L >= 64 and divisible by 32, within budget")
            groups.setdefault(frame.shape[1], []).append(index)
        values = frames[0].new_empty(len(frames), 512)
        for length, indices in groups.items():
            batch_size = self.max_batch_samples // length
            for start in range(0, len(indices), batch_size):
                current = indices[start:start+batch_size]
                output = self.core(torch.stack([frames[i] for i in current]))
                if output.shape != (len(current), 512) or not torch.isfinite(output).all():
                    raise ValueError("Pinned Mantis combined output must have shape [G,512] and be finite")
                values[current] = output
        return EncodedSignal(values, intervals)
