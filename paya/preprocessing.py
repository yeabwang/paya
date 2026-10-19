"""Clean native signals, then format only the encoder selected by the router."""

from dataclasses import dataclass, asdict, replace
import hashlib
import json

import neurokit2 as nk
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
import torch

from .inputs import Channel, UnsupportedSignalError, validate_state
from .physiology import clean_channel, check_eeg_channels
from .provenance import PREPARATION_CONTRACT, NATIVE_CONTRACT

MANTIS_RATES = {"ecg": 500, "eeg": 500, "accelerometer": 100, "gyroscope": 100,
                "ppg": 125, "eda": 32, "respiration": 32, "temperature": 32, "emg": 2000}


@dataclass(frozen=True)
class PreparationConfig:
    max_channels: int = 64
    max_samples: int = 1_000_000
    max_normwear_channels: int = 4
    max_mantis_frames: int = 640
    max_signal_tokens: int = 1024
    time_bins: int = 8
    filter_order: int = 4

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Preparation budgets/settings must be positive integers")
        if self.time_bins > 43:
            raise ValueError("NormWear has only 43 temporal patch rows per frame")


@dataclass
class EncoderInput:
    frames: torch.Tensor             # [real channel/frame pairs, 1, samples]
    bounds: torch.Tensor             # [pairs, 2], relative interval boundaries
    channels: int = 1                # NormWear grouping only

    def to(self, device):
        return EncoderInput(self.frames.to(device), self.bounds.to(device), self.channels)


@dataclass
class CleanedState:
    channels: list[Channel]          # One cleaned array per physical channel, at its native rate
    compatible: torch.Tensor        # [2], deterministic encoder eligibility
    identity: str
    audit: dict

    def to(self, device):
        return CleanedState(self.channels, self.compatible.to(device), self.identity, self.audit)


def frame_starts(length, frame, stride):
    if length < frame:
        return []
    starts = list(range(0, length-frame+1, stride))
    if starts[-1] != length-frame:
        starts.append(length-frame)
    return starts


class SignalPreprocessor:
    """Retain one native-rate cleaned signal and format it after encoder selection."""

    def __init__(self, config=None):
        self.config = config or PreparationConfig()

    def identity(self):
        return {"contract": PREPARATION_CONTRACT, "config": asdict(self.config),
                "native_processing": NATIVE_CONTRACT, "routing_input": "native-cleaned-channels",
                "encoder_formatting": "after-selection-from-native-cleaned-signal",
                "neurokit2": nk.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "resampling": "centered-polyphase-linear-boundary", "mantis_rates": MANTIS_RATES,
                "normwear_rate": 65, "normwear_samples": 390,
                "normwear_gaussian_sigma": 1.3, "anti_alias_fraction": 0.4}

    def _check_encoder(self, channels, expert):
        # Eligibility uses dimensions and metadata, without resampling or making frames.
        if expert == 0:
            if len(channels) > self.config.max_normwear_channels:
                raise UnsupportedSignalError("NormWear channel budget exceeded")
            if any(c.sensor_type in ("temperature", "emg") for c in channels):
                raise UnsupportedSignalError("No NormWear temperature/raw-EMG formatting is enabled")
            lengths = [round(len(c.samples)*65/c.rate) for c in channels]
            if len(set(lengths)) != 1:
                raise UnsupportedSignalError("NormWear channels do not align on the 65 Hz grid")
            if lengths[0] < 390:
                raise UnsupportedSignalError("NormWear needs six seconds of real coverage")
            count = ((lengths[0]-390+389)//390+1)*len(channels)
            if count*(1+13*self.config.time_bins) > self.config.max_signal_tokens:
                raise UnsupportedSignalError("NormWear signal-token budget exceeded")
        else:
            count = 0
            for channel in channels:
                rate = MANTIS_RATES[channel.sensor_type]
                length = round(len(channel.samples)*rate/channel.rate)
                if length < 512:
                    raise UnsupportedSignalError(f"Mantis {channel.sensor_type} needs {512/rate:g} seconds of real coverage")
                count += (length-512+255)//256+1
                if count > min(self.config.max_mantis_frames, self.config.max_signal_tokens):
                    raise UnsupportedSignalError("Mantis frame/token budget exceeded")

    def clean(self, state):
        """Validate and clean native channels; compute eligibility without encoder formatting."""
        channels = validate_state(state, self.config.max_channels, self.config.max_samples)
        duration = len(channels[0].samples) / channels[0].rate
        if not np.isfinite(duration) or duration <= 0:
            raise UnsupportedSignalError("Unrepresentable physical duration")
        max_duration = max(self.config.max_signal_tokens*6,
                           (self.config.max_mantis_frames+1)*256/min(MANTIS_RATES.values()))
        if duration > max_duration:
            raise UnsupportedSignalError("Physical duration exceeds every encoder token budget")
        compatible, reasons = [], {}
        for expert, name in enumerate(("normwear", "mantis")):
            try:
                self._check_encoder(channels, expert)
                compatible.append(True)
            except UnsupportedSignalError as error:
                compatible.append(False)
                reasons[name] = str(error)
        if not any(compatible):
            raise UnsupportedSignalError(f"No compatible encoder preparation: {reasons}")

        cleaned, quality = [], []
        for channel in channels:
            values, observations = clean_channel(channel)
            values = np.asarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise UnsupportedSignalError("Native cleaning overflowed the waveform representation")
            cleaned.append(replace(channel, samples=values))
            quality.append(observations)
        check_eeg_channels(cleaned, [c.samples for c in cleaned], quality)
        audit = {"duration_s": duration, "channels": len(cleaned), "preparation": self.identity(),
                 "quality": quality, "unavailable": reasons}
        digest = hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode())
        for channel in cleaned:
            digest.update(json.dumps(channel.metadata, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode())
            digest.update(channel.samples.tobytes())
        return CleanedState(cleaned, torch.tensor(compatible, dtype=torch.bool), digest.hexdigest(), audit)

    def _convert(self, values, native_rate, target):
        clean = values
        if target < native_rate:
            clean = nk.signal_filter(clean, sampling_rate=native_rate, highcut=0.4*target,
                                     method="butterworth", order=self.config.filter_order)
        length = round(len(clean)*target/native_rate)
        # Centering preserves constant levels; linear extension avoids artificial boundary steps.
        offset = clean.mean()
        result = resample_poly(clean-offset, length, len(clean), padtype="line")+offset if length != len(clean) else clean
        result = np.asarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise UnsupportedSignalError("Nonfinite values after encoder formatting")
        return result

    def format_for_encoder(self, cleaned, expert):
        """Resample and frame the cleaned signal for exactly one selected encoder."""
        if isinstance(expert, bool) or not isinstance(expert, int) or expert not in (0, 1):
            raise ValueError("Encoder index must be 0 or 1")
        if cleaned.audit.get("preparation") != self.identity():
            raise ValueError("Cleaned signal does not match current physiological preprocessing")
        if not cleaned.compatible[expert]:
            raise UnsupportedSignalError("Cannot format an incompatible encoder")
        channels = cleaned.channels
        self._check_encoder(channels, expert)
        duration = len(channels[0].samples)/channels[0].rate
        frames, bounds = [], []
        if expert == 0:
            arrays = [self._convert(c.samples, c.rate, 65) for c in channels]
            for start in frame_starts(len(arrays[0]), 390, 390):
                for array in arrays:
                    frames.append(gaussian_filter1d(nk.signal_detrend(array[start:start+390], order=1), 1.3))
                    bounds.append([start/65/duration, (start+390)/65/duration])
        else:
            for channel in channels:
                rate = MANTIS_RATES[channel.sensor_type]
                array = self._convert(channel.samples, channel.rate, rate)
                for start in frame_starts(len(array), 512, 256):
                    frames.append(array[start:start+512])
                    bounds.append([start/rate/duration, (start+512)/rate/duration])
        return EncoderInput(torch.from_numpy(np.stack(frames).astype(np.float32))[:, None],
                            torch.tensor(bounds, dtype=torch.float32), len(channels) if expert == 0 else 1)
