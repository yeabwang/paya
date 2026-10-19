"""Validate native signals, clean raw channels and form Mantis windows."""

from dataclasses import dataclass, asdict, replace
import hashlib
import json

import neurokit2 as nk
import numpy as np
import torch

from .inputs import Channel, UnsupportedSignalError, validate_state
from .physiology import clean_channel, check_eeg_channels
from .provenance import PREPARATION_CONTRACT, NATIVE_CONTRACT


@dataclass(frozen=True)
class PreparationConfig:
    max_channels: int = 64
    max_samples: int = 1_000_000
    max_signal_tokens: int = 128

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Preparation budgets must be positive integers")
        if self.max_samples < 64:
            raise ValueError("Sample budget must allow at least 64 native samples")


@dataclass
class EncoderInput:
    frames: list[torch.Tensor]       # Each native window has shape [1,L], L divisible by 32
    intervals_s: torch.Tensor        # [windows, 2], elapsed start/end seconds

    def to(self, device):
        return EncoderInput([frame.to(device) for frame in self.frames], self.intervals_s.to(device))


@dataclass
class PreparedState:
    channels: list[Channel]          # Samples retain their declared native rates
    identity: str
    audit: dict


def native_windows(length):
    """Cover all samples with one aligned window or a prefix/suffix pair."""
    if length < 64:
        raise UnsupportedSignalError("Mantis needs 64 native samples: two per statistics patch")
    aligned = length // 32 * 32
    return [(0, aligned)] if aligned == length else [(0, aligned), (length-aligned, length)]


class SignalPreprocessor:
    """Bypass NeuroKit for processed channels and preserve every native sampling rate."""

    def __init__(self, config=None):
        self.config = config or PreparationConfig()

    def identity(self):
        return {"contract": PREPARATION_CONTRACT, "config": asdict(self.config),
                "native_processing": NATIVE_CONTRACT, "neurokit2": nk.__version__,
                "numpy": np.__version__, "signal_condition": ["raw", "processed"],
                "sampling": "native-no-resampling", "framing": "full-context-prefix-suffix-32",
                "minimum_samples": 64, "intervals": "elapsed-start-end-seconds",
                "encoder_batch_sample_budget": self.config.max_samples}

    def _check_frames(self, channels):
        count = 0
        for channel in channels:
            count += len(native_windows(len(channel.samples)))
        if count > self.config.max_signal_tokens:
            raise UnsupportedSignalError("Mantis signal-token budget exceeded")

    def prepare(self, state):
        """Validate all channels and run NeuroKit only on those marked raw."""
        channels = validate_state(state, self.config.max_channels, self.config.max_samples)
        self._check_frames(channels)
        prepared, quality, raw_channels, raw_audits = [], [], [], []
        for channel in channels:
            if channel.metadata["signal_condition"] == "raw":
                values, observations = clean_channel(channel)
                values = np.asarray(values, dtype=np.float32)
                if values.shape != channel.samples.shape or not np.isfinite(values).all():
                    raise UnsupportedSignalError("Native cleaning changed sample count or produced nonfinite values")
                current = replace(channel, samples=values)
                observations["signal_condition"] = "raw"
                raw_channels.append(current)
                raw_audits.append(observations)
            else:
                current = channel
                observations = {"sensor_type": channel.sensor_type, "sampling_rate_hz": channel.rate,
                                "signal_condition": "processed", "cleaning": "skipped"}
            prepared.append(current)
            quality.append(observations)
        if raw_channels:
            check_eeg_channels(raw_channels, [c.samples for c in raw_channels], raw_audits)
        audit = {"duration_s": channels[0].metadata["duration_s"], "channels": len(prepared),
                 "preparation": self.identity(), "quality": quality}
        digest = hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode())
        for channel in prepared:
            digest.update(json.dumps(channel.metadata, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode())
            digest.update(channel.samples.tobytes())
        return PreparedState(prepared, digest.hexdigest(), audit)

    def format(self, prepared):
        """Retain full context and sample coverage without resampling or padding."""
        if prepared.audit.get("preparation") != self.identity():
            raise ValueError("Prepared signal does not match current preprocessing")
        self._check_frames(prepared.channels)
        frames, intervals = [], []
        for channel in prepared.channels:
            for start, end in native_windows(len(channel.samples)):
                frames.append(torch.from_numpy(channel.samples[start:end])[None])
                intervals.append([start/channel.rate, end/channel.rate])
        with np.errstate(over="ignore", invalid="ignore"):
            intervals = torch.from_numpy(np.asarray(intervals, dtype=np.float32))
        if (not torch.isfinite(intervals).all() or (intervals[:, 0] < 0).any()
                or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise UnsupportedSignalError("Recording intervals cannot be represented in float32 seconds")
        return EncoderInput(frames, intervals)
