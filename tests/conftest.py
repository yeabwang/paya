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


def state(kind="ecg", rate=500, duration=6, unit=None):
    units = {"ecg": "mV", "eeg": "uV", "accelerometer": "m/s2", "gyroscope": "rad/s",
             "ppg": "au", "eda": "uS", "respiration": "au", "temperature": "C", "emg": "uV"}
    t = np.arange(round(rate*duration))/rate
    waveform = np.sin(2*np.pi*min(1.3, rate/8)*t) + 0.05*np.cos(2*np.pi*min(7, rate/4)*t)
    if kind == "temperature":
        waveform = 37 + 0.01*t
    return {"signals": [{"samples": waveform.tolist(), "metadata": {
        "sensor_type": kind, "sampling_rate_hz": rate, "unit": unit or units[kind]}}]}


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
        values = self.base(encoder_input.frames.mean(-1))
        return EncodedSignal(values, encoder_input.bounds.mean(-1))


def new_model():
    torch.manual_seed(100)
    return PayaModel(EncoderDouble(768), EncoderDouble(512), TextDouble(),
                     head={"width": 256, "heads": 4, "layers": 2, "dropout": 0}).eval()


@pytest.fixture
def model():
    return new_model()
