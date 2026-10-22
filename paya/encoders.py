"""Frozen SPOTR global, channel and temporal tokens with their validity mask."""

import torch
from torch import nn

from .vendor import signal


class SignalEncoder(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core.requires_grad_(False).eval()

    @classmethod
    def load(cls, directory):
        return cls(signal.load_encoder(directory))

    @torch.no_grad()
    def forward(self, encoder_input):
        return signal.encode(self.core, encoder_input.samples)
