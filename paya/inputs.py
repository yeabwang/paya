"""Validate native signal samples, required metadata and separate text inputs."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .vendor import signal

TYPES = ("eeg", "ieeg", "ecg", "ppg")
SIGNAL_CONDITIONS = ("raw", "processed")


class UnsupportedSignalError(ValueError):
    """The supplied measurements cannot satisfy a supported preparation."""


@dataclass(frozen=True)
class Channel:
    samples: np.ndarray
    sensor_type: str
    rate: float
    start: float
    metadata: dict


def positive_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not a string or boolean")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_text(question, criteria):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 64:
        raise ValueError("criteria must be an ordered list of 2–64 strings")
    if any(not isinstance(c, str) or not c.strip() for c in criteria):
        raise ValueError("criteria entries must be nonempty strings")
    if len({c.strip() for c in criteria}) != len(criteria):
        raise ValueError("criteria entries must be distinct")


def validate_state(state, max_channels=signal.MAX_CHANNELS, max_samples=signal.MAX_SAMPLES):
    if not isinstance(state, Mapping) or not isinstance(state.get("signals"), list):
        raise ValueError("state must contain a signals list")
    if not 1 <= len(state["signals"]) <= max_channels:
        raise ValueError(f"Supply 1–{max_channels} physical channels")
    channels = []
    for entry in state["signals"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("metadata"), Mapping):
            raise ValueError("Each signal needs samples and a metadata object")
        meta = dict(entry["metadata"])
        kind = meta.get("sensor_type")
        if not isinstance(kind, str) or kind.strip().lower() not in TYPES:
            raise ValueError(f"metadata.sensor_type must be one of {TYPES}")
        kind = kind.strip().lower()
        rate = positive_number(meta.get("sampling_rate_hz"), "metadata.sampling_rate_hz")
        duration = positive_number(meta.get("duration_s"), "metadata.duration_s")
        condition = meta.get("signal_condition", "raw")
        if not isinstance(condition, str) or condition not in SIGNAL_CONDITIONS:
            raise ValueError("metadata.signal_condition must be raw or processed")
        # Units and lead labels are outside the input contract; values retain their supplied scale.
        meta.pop("unit", None)
        meta.pop("lead", None)
        if "powerline_hz" in meta:
            powerline = meta["powerline_hz"]
            if isinstance(powerline, (bool, np.bool_)) or not isinstance(powerline, Real) or powerline not in (0, 50, 60):
                raise ValueError("metadata.powerline_hz must be numeric 0, 50, or 60")
        raw = np.asarray(entry.get("samples"))
        if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not signal.MIN_SAMPLES <= raw.size <= max_samples:
            raise ValueError(f"samples must contain {signal.MIN_SAMPLES}–{max_samples} real numeric measurements")
        values = raw.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Samples must be finite; missing intervals cannot be filled silently")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Samples exceed the float32 waveform representation")
        start = meta.get("start_time_s", 0.0)
        if isinstance(start, (bool, np.bool_)) or not isinstance(start, Real) or not math.isfinite(start):
            raise ValueError("metadata.start_time_s must be finite")
        measured_duration = len(values)/rate
        if not math.isfinite(measured_duration) or not math.isclose(
                duration, measured_duration, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("metadata.duration_s disagrees with samples and sampling rate")
        if "timestamps_s" in meta:
            ts = np.asarray(meta["timestamps_s"], dtype=np.float64)
            if ts.shape != values.shape or not np.isfinite(ts).all():
                raise ValueError("Timestamps must be finite and match samples")
            if not np.allclose(np.diff(ts), 1/rate, rtol=0.01, atol=1e-9):
                raise ValueError("Timestamps disagree with the declared uniform sampling rate")
            if not math.isclose(ts[0], start, rel_tol=0, abs_tol=max(1e-9, 0.01/rate)):
                raise ValueError("Timestamps disagree with start_time_s")
        for field in ("placement", "reference"):
            if field in meta and (not isinstance(meta[field], str) or not meta[field].strip()):
                raise ValueError(f"metadata.{field} must be a nonempty string when supplied")
        meta.update(sensor_type=kind, sampling_rate_hz=rate, duration_s=duration, signal_condition=condition)
        channels.append(Channel(values, kind, rate, float(start), meta))
    first = channels[0]
    for c in channels[1:]:
        if c.rate != first.rate or len(c.samples) != len(first.samples):
            raise ValueError("All channels must have the same sampling rate and sample count for one encoder input")
        if abs(c.start-first.start) > max(1e-9, 0.01/first.rate):
            raise ValueError("Channels must have the same start_time_s")
    return channels
