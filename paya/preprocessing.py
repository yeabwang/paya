"""Deterministic NeuroKit preparation, completed before the learned router."""

from dataclasses import dataclass, asdict
import hashlib
import json

import neurokit2 as nk
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
import torch

from .inputs import UnsupportedSignalError, validate_state

PREPARATION_CONTRACT = "neurokit-two-views-v2"
MANTIS_RATES = {"ecg": 500, "eeg": 500, "accelerometer": 100, "gyroscope": 100,
                "ppg": 125, "eda": 32, "respiration": 32, "temperature": 32, "emg": 2000}


@dataclass(frozen=True)
class PreparationConfig:
    max_channels: int = 64
    max_samples: int = 1_000_000
    max_normwear_channels: int = 4
    max_mantis_frames: int = 128
    max_signal_tokens: int = 1024
    time_bins: int = 8
    filter_order: int = 4

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Preparation budgets/settings must be positive integers")
        if self.time_bins > 43:
            raise ValueError("NormWear has only 43 temporal patch rows per frame")


@dataclass
class PreparedView:
    frames: torch.Tensor             # [real channel/frame pairs, 1, samples]
    bounds: torch.Tensor             # [pairs, 2], relative prepared interval boundaries
    channels: int = 1                # NormWear grouping only; never a head feature

    def to(self, device):
        return PreparedView(self.frames.to(device), self.bounds.to(device), self.channels)


@dataclass
class PreparedState:
    views: tuple[PreparedView | None, PreparedView | None]
    identity: str
    audit: dict

    @property
    def compatible(self):
        return torch.tensor([v is not None for v in self.views], dtype=torch.bool)

    def to(self, device):
        return PreparedState(tuple(v.to(device) if v is not None else None for v in self.views),
                             self.identity, self.audit)


def frame_starts(length, frame, stride):
    if length < frame:
        return []
    starts = list(range(0, length-frame+1, stride))
    if starts[-1] != length-frame:
        starts.append(length-frame)
    return starts


class SignalPreprocessor:
    def __init__(self, config=None):
        self.config = config or PreparationConfig()

    def identity(self):
        return {"contract": PREPARATION_CONTRACT, "config": asdict(self.config),
                "neurokit2": nk.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "resampling": "polyphase-linear-boundary", "mantis_rates": MANTIS_RATES,
                "normwear_rate": 65, "normwear_samples": 390,
                "normwear_gaussian_sigma": 1.3, "anti_alias_fraction": 0.4}

    def _convert(self, channel, target):
        # This low-pass preserves DC; task-specific high-pass cleaning is not a default.
        cutoff = 0.4 * min(channel.rate, target)
        clean = nk.signal_filter(channel.samples.astype(np.float64), sampling_rate=channel.rate,
                                 highcut=cutoff, method="butterworth", order=self.config.filter_order)
        length = round(len(clean)*target/channel.rate)
        # NeuroKit's poly method calls this SciPy operation with implicit zero
        # boundary padding. Linear boundary extension preserves physical DC levels
        # (e.g. temperature) without extending the returned measurement interval.
        result = resample_poly(clean, length, len(clean), padtype="line") if length != len(clean) else clean
        result = np.asarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise UnsupportedSignalError("Nonfinite values after NeuroKit preparation")
        return result

    def prepare(self, state):
        channels = validate_state(state, self.config.max_channels, self.config.max_samples)
        digest = hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode())
        for c in channels:
            digest.update(json.dumps(c.metadata, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode())
            digest.update(c.samples.tobytes())
        duration = len(channels[0].samples) / channels[0].rate
        if not np.isfinite(duration) or duration <= 0:
            raise UnsupportedSignalError("Unrepresentable physical duration")
        # Even the slowest configured view cannot fit a longer real interval in
        # the token budget. Check before multiplying a pathological duration/rate.
        max_duration = max(self.config.max_signal_tokens*6,
                           (self.config.max_mantis_frames+1)*256/min(MANTIS_RATES.values()))
        if duration > max_duration:
            raise UnsupportedSignalError("Physical duration exceeds every encoder token budget")
        reasons, audit = {}, {"duration_s": duration, "channels": len(channels), "preparation": self.identity()}
        audit["quality"] = [{"flatline": bool(np.ptp(c.samples.astype(np.float64)) == 0),
                             "minimum": float(c.samples.min()), "maximum": float(c.samples.max())}
                            for c in channels]

        def normwear_view():
            if len(channels) > self.config.max_normwear_channels:
                raise UnsupportedSignalError("NormWear channel budget exceeded")
            if any(c.sensor_type in ("temperature", "emg") for c in channels):
                raise UnsupportedSignalError("No NormWear temperature/raw-EMG preparation is enabled")
            lengths = [round(len(c.samples)*65/c.rate) for c in channels]
            if len(set(lengths)) != 1:
                raise UnsupportedSignalError("NormWear channels do not align on the 65 Hz grid")
            if lengths[0] < 390:
                raise UnsupportedSignalError("NormWear needs six seconds of real coverage")
            count = ((lengths[0]-390+389)//390+1)*len(channels)
            if count*(1+13*self.config.time_bins) > self.config.max_signal_tokens:
                raise UnsupportedSignalError("NormWear signal-token budget exceeded")
            starts = frame_starts(lengths[0], 390, 390)
            arrays = [gaussian_filter1d(nk.signal_detrend(self._convert(c, 65), order=1), 1.3)
                      for c in channels]
            values, bounds = [], []
            for start in starts:
                for array in arrays:
                    values.append(array[start:start+390])
                    bounds.append([start/65/duration, (start+390)/65/duration])
            return PreparedView(torch.from_numpy(np.stack(values).astype(np.float32))[:, None],
                                torch.tensor(bounds, dtype=torch.float32), len(channels))

        def mantis_view():
            specifications, count = [], 0
            for c in channels:
                rate = MANTIS_RATES[c.sensor_type]
                length = round(len(c.samples)*rate/c.rate)
                # Bound the prospective frame count before allocating an upsampled array.
                estimated = max(0, (length-512+255)//256+1) if length >= 512 else 0
                count += estimated
                if count > min(self.config.max_mantis_frames, self.config.max_signal_tokens):
                    raise UnsupportedSignalError("Mantis frame/token budget exceeded")
                starts = frame_starts(length, 512, 256)
                if not starts:
                    raise UnsupportedSignalError(f"Mantis {c.sensor_type} needs {512/rate:g} seconds of real coverage")
                specifications.append((c, rate, starts))
            values, bounds = [], []
            for c, rate, starts in specifications:
                array = self._convert(c, rate)
                for start in starts:
                    values.append(array[start:start+512])
                    bounds.append([start/rate/duration, (start+512)/rate/duration])
            return PreparedView(torch.from_numpy(np.stack(values))[:, None],
                                torch.tensor(bounds, dtype=torch.float32))

        views = []
        for name, build in (("normwear", normwear_view), ("mantis", mantis_view)):
            try:
                views.append(build())
            except (ValueError, UnsupportedSignalError) as error:
                views.append(None)
                reasons[name] = str(error)
        if all(v is None for v in views):
            raise UnsupportedSignalError(f"No compatible encoder preparation: {reasons}")
        audit["unavailable"] = reasons
        return PreparedState(tuple(views), digest.hexdigest(), audit)
