"""Validate and clean one aligned native-rate recording for SPOTR."""

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
    max_samples: int = SPOTR_MAX_SAMPLES

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Preparation budgets must be positive integers")
        if self.max_channels > SPOTR_MAX_CHANNELS:
            raise ValueError("SPOTR permits at most 128 channels")
        if not 2 <= self.max_samples <= SPOTR_MAX_SAMPLES:
            raise ValueError("SPOTR sample budget must be between 2 and 12800")


@dataclass
class EncoderInput:
    samples: torch.Tensor           # [C,L], the complete aligned recording
    intervals_s: torch.Tensor        # [1,2], elapsed start/end seconds

    def to(self, device):
        return EncoderInput(self.samples.to(device), self.intervals_s.to(device))


@dataclass
class PreparedState:
    channels: list[Channel]          # Samples retain their declared native rates
    identity: str
    audit: dict


class SignalPreprocessor:
    """Bypass NeuroKit for processed channels and preserve every native sampling rate."""

    def __init__(self, config=None):
        self.config = config or PreparationConfig()

    def identity(self):
        return {"contract": PREPARATION_CONTRACT, "config": asdict(self.config),
                "native_processing": NATIVE_CONTRACT, "neurokit2": nk.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__, "signal_condition": ["raw", "processed"],
                "sampling": "native-no-resampling", "format": "one-aligned-recording",
                "channel_order": "as-supplied", "minimum_samples": 2,
                "encoder_max_channels": SPOTR_MAX_CHANNELS, "encoder_max_samples": SPOTR_MAX_SAMPLES,
                "encoder_normalization": "channel-zscore", "encoder_padding": "zero-to-multiple-of-200",
                "intervals": "elapsed-start-end-seconds"}

    def prepare(self, state):
        """Validate all channels and run NeuroKit only on those marked raw."""
        channels = validate_state(state, self.config.max_channels, self.config.max_samples)
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
        """Stack the complete recording in supplied channel order without external padding."""
        if prepared.audit.get("preparation") != self.identity():
            raise ValueError("Prepared signal does not match current preprocessing")
        samples = torch.from_numpy(np.stack([c.samples for c in prepared.channels]))
        duration = len(prepared.channels[0].samples)/prepared.channels[0].rate
        with np.errstate(over="ignore", invalid="ignore"):
            intervals = torch.from_numpy(np.asarray([[0, duration]], dtype=np.float32))
        if not torch.isfinite(intervals).all() or intervals[0, 1] <= 0:
            raise UnsupportedSignalError("Recording duration cannot be represented in float32 seconds")
        return EncoderInput(samples, intervals)
