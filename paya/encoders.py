"""Frozen encoder adapters and parameter-free output-width formatting."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from .provenance import NORMWEAR_ID, NORMWEAR_REVISION, MANTIS_ID, MANTIS_REVISION, verify_source
from .vendor.normwear import NormWear


@dataclass
class EncodedSignal:
    values: torch.Tensor             # [M, encoder width]
    times: torch.Tensor              # [M], relative times within the measured interval


class FixedWidthAdapter(nn.Module):
    def __init__(self, input_width):
        super().__init__()
        if input_width not in (512, 768):
            raise ValueError("Only the two selected encoder widths are supported")
        self.input_width = input_width

    def forward(self, values):
        if values.shape[-1] != self.input_width or not torch.isfinite(values).all():
            raise ValueError("Invalid encoder output")
        return F.pad(values, (0, 768-self.input_width)) if self.input_width == 512 else values


class NormWearEncoder(nn.Module):
    width = 768

    def __init__(self, core, time_bins=8):
        super().__init__()
        self.core = core.requires_grad_(False).eval()
        self.time_bins = time_bins

    @classmethod
    def load(cls, directory, time_bins=8):
        directory = Path(directory)
        verify_source(directory, NORMWEAR_ID, NORMWEAR_REVISION)
        core = NormWear(**json.loads((directory / "config.json").read_text()))
        core.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return cls(core, time_bins)

    @torch.no_grad()
    def forward(self, encoder_input):
        pairs = encoder_input.frames.shape[0]
        if pairs % encoder_input.channels or encoder_input.frames.shape[1:] != (1, 390):
            raise ValueError("Invalid NormWear frame grouping")
        windows = pairs // encoder_input.channels
        patches = self.core(encoder_input.frames.reshape(windows, encoder_input.channels, 390))
        if patches.shape != (windows, encoder_input.channels, 560, 768):
            raise ValueError("Pinned NormWear patch shape differs from its contract")
        flat = patches.reshape(pairs, 560, 768)
        grid = flat[:, 1:].reshape(pairs, 43, 13, 768).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid, (self.time_bins, 13)).permute(0, 2, 3, 1)
        values = torch.cat((flat[:, :1], pooled.reshape(pairs, -1, 768)), dim=1)
        count = values.shape[1]
        times = values.new_empty(pairs, count)
        bounds = encoder_input.bounds
        times[:, 0] = bounds.mean(-1)
        fractions = (torch.arange(self.time_bins, device=values.device)+0.5)/self.time_bins
        centers = bounds[:, :1] + (bounds[:, 1:]-bounds[:, :1])*fractions[None]
        times[:, 1:] = centers.repeat_interleave(13, dim=-1)
        return EncodedSignal(values.reshape(-1, 768), times.reshape(-1))


class MantisEncoder(nn.Module):
    width = 512

    def __init__(self, core):
        super().__init__()
        self.core = core.requires_grad_(False).eval()

    @classmethod
    def load(cls, directory):
        from mantis.architecture import MantisV2
        verify_source(directory, MANTIS_ID, MANTIS_REVISION)
        core = MantisV2(device="cpu", return_transf_layer=2, output_token="combined", pre_training=False)
        core = core.from_pretrained(str(directory), local_files_only=True)
        core.remove_transf_layers()
        return cls(core)

    @torch.no_grad()
    def forward(self, encoder_input):
        values = self.core(encoder_input.frames)
        if values.shape != (encoder_input.frames.shape[0], 512) or not torch.isfinite(values).all():
            raise ValueError("Pinned Mantis combined output must have shape [R,512] and be finite")
        return EncodedSignal(values, encoder_input.bounds.mean(-1))
