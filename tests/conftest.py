"""Small deterministic doubles exercise wiring, never predictive performance."""

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from paya.encoders import EncodedSignal
from paya.model import PayaModel


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def state(kind="ecg", rate=500, duration=6, condition=None):
    t = np.arange(round(rate*duration))/rate
    waveform = np.sin(2*np.pi*min(1.3, rate/8)*t) + 0.05*np.cos(2*np.pi*min(7, rate/4)*t)
    metadata = {"sensor_type": kind, "sampling_rate_hz": rate, "duration_s": len(t)/rate}
    if condition is not None:
        metadata["signal_condition"] = condition
    return {"signals": [{"samples": waveform.tolist(), "metadata": metadata}]}


class TextDouble:
    model = None

    def __init__(self):
        self.calls = []

    def encode(self, strings):
        self.calls.append(list(strings))
        vectors = []
        for text in strings:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vector = torch.randn(768, generator=torch.Generator().manual_seed(seed))
            vectors.append(vector/vector.norm())
        return torch.stack(vectors)


class EncoderDouble(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.base = nn.Linear(1, width)
        self.calls = 0

    def forward(self, encoder_input):
        self.calls += 1
        values = self.base(encoder_input.samples.mean().reshape(1, 1))
        return EncodedSignal(values, encoder_input.intervals_s)


def new_model():
    torch.manual_seed(100)
    return PayaModel(EncoderDouble(512), TextDouble(),
                     head={"width": 256, "heads": 4, "layers": 2, "dropout": 0}).eval()


@pytest.fixture
def model():
    return new_model()
