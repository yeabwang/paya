"""Validate native signals, clean raw channels and form SPOTR channel groups and windows."""

from dataclasses import dataclass, asdict, replace
import hashlib
import json

import neurokit2 as nk
import numpy as np
import scipy
import torch

from .inputs import Channel, UnsupportedSignalError, validate_state
from .physiology import clean_channel, check_eeg_channels
from .provenance import PREPARATION_CONTRACT, NATIVE_CONTRACT, SPOTR_MAX_CHANNELS, SPOTR_MAX_SAMPLES


@dataclass(frozen=True)
class PreparationConfig:
    max_channels: int = SPOTR_MAX_CHANNELS
    max_samples: int = 1_000_000
    max_signal_tokens: int = 128

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Preparation budgets must be positive integers")
        if self.max_channels > SPOTR_MAX_CHANNELS or self.max_samples < 2:
            raise ValueError("SPOTR permits at most 128 channels and needs at least two samples")


@dataclass
class EncoderInput:
    frames: list[torch.Tensor]       # Each aligned native window has shape [C,L]
    intervals_s: torch.Tensor        # [windows, 2], elapsed start/end seconds

    def to(self, device):
        return EncoderInput([frame.to(device) for frame in self.frames], self.intervals_s.to(device))


@dataclass
class PreparedState:
    channels: list[Channel]          # Samples retain their declared native rates
    identity: str
    audit: dict


def spotr_windows(length):
    """Cover native samples within the pretrained temporal-position capacity."""
    if length < 2:
        raise UnsupportedSignalError("SPOTR needs at least two native samples for channel normalization")
    if length <= SPOTR_MAX_SAMPLES:
        return [(0, length)]
    starts = list(range(0, length-SPOTR_MAX_SAMPLES+1, SPOTR_MAX_SAMPLES))
    if starts[-1] != length-SPOTR_MAX_SAMPLES:
        starts.append(length-SPOTR_MAX_SAMPLES)
    return [(start, start+SPOTR_MAX_SAMPLES) for start in starts]


def channel_groups(channels):
    """Group aligned channels by native rate and length, preserving their order."""
    groups = {}
    for channel in channels:
        groups.setdefault((channel.rate, len(channel.samples)), []).append(channel)
    return list(groups.values())


class SignalPreprocessor:
    """Bypass NeuroKit for processed channels and preserve every native sampling rate."""

    def __init__(self, config=None):
        self.config = config or PreparationConfig()

    def identity(self):
        return {"contract": PREPARATION_CONTRACT, "config": asdict(self.config),
                "native_processing": NATIVE_CONTRACT, "neurokit2": nk.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__, "signal_condition": ["raw", "processed"],
                "sampling": "native-no-resampling", "framing": "aligned-groups-capacity-windows",
                "channel_order": "as-supplied", "minimum_samples": 2,
                "encoder_max_channels": SPOTR_MAX_CHANNELS, "encoder_max_samples": SPOTR_MAX_SAMPLES,
                "encoder_normalization": "channel-zscore", "encoder_padding": "zero-to-multiple-of-200",
                "intervals": "elapsed-start-end-seconds"}

    def _check_frames(self, channels):
        count = 0
        for group in channel_groups(channels):
            count += len(spotr_windows(len(group[0].samples)))
        if count > self.config.max_signal_tokens:
            raise UnsupportedSignalError("SPOTR signal-token budget exceeded")

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
        """Group native channels and slice capacity-limited windows without external padding."""
        if prepared.audit.get("preparation") != self.identity():
            raise ValueError("Prepared signal does not match current preprocessing")
        self._check_frames(prepared.channels)
        frames, intervals = [], []
        for group in channel_groups(prepared.channels):
            rate = group[0].rate
            for start, end in spotr_windows(len(group[0].samples)):
                frames.append(torch.from_numpy(np.stack([c.samples[start:end] for c in group])))
                intervals.append([start/rate, end/rate])
        with np.errstate(over="ignore", invalid="ignore"):
            intervals = torch.from_numpy(np.asarray(intervals, dtype=np.float32))
        if (not torch.isfinite(intervals).all() or (intervals[:, 0] < 0).any()
                or (intervals[:, 1] <= intervals[:, 0]).any()):
            raise UnsupportedSignalError("Recording intervals cannot be represented in float32 seconds")
        return EncoderInput(frames, intervals)
