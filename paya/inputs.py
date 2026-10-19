"""Validate signal metadata, physical units and separate question/criteria inputs."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

ENDING = "The answer is:"
TYPES = ("ecg", "eeg", "accelerometer", "gyroscope", "ppg", "eda",
         "respiration", "temperature", "emg")


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


def canonical_unit(samples, kind, unit):
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("metadata.unit must specify the measurement convention")
    unit = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    electrical = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
    if kind in ("ecg", "eeg", "emg") and unit in electrical:
        target = "mv" if kind == "ecg" else "uv"
        return samples * electrical[unit] / electrical[target], target
    conversions = {
        "accelerometer": {"g": (9.80665, 0, "m/s2"), "m/s2": (1, 0, "m/s2"),
                          "m/s²": (1, 0, "m/s2")},
        "gyroscope": {"rad/s": (1, 0, "rad/s"), "deg/s": (math.pi/180, 0, "rad/s")},
        "eda": {"us": (1, 0, "us"), "ms": (1000, 0, "us"), "s": (1e6, 0, "us")},
        "temperature": {"c": (1, 0, "c"), "°c": (1, 0, "c"),
                        "f": (5/9, -32*5/9, "c"), "°f": (5/9, -32*5/9, "c"),
                        "k": (1, -273.15, "c")},
        "ppg": {u: (1, 0, u) for u in ("au", "counts", "normalized")},
        "respiration": {**{u: (1, 0, u) for u in ("au", "normalized", "l")},
                        "ml": (0.001, 0, "l")},
    }
    if unit not in conversions.get(kind, {}):
        raise ValueError(f"Unsupported unit {unit!r} for {kind}; physical units cannot be guessed")
    scale, offset, target = conversions[kind][unit]
    return samples * scale + offset, target


def validate_state(state, max_channels=64, max_samples=1_000_000):
    if not isinstance(state, Mapping) or not isinstance(state.get("signals"), list):
        raise ValueError("state must contain a signals list")
    if not 1 <= len(state["signals"]) <= max_channels:
        raise ValueError(f"Supply 1–{max_channels} physical channels")
    channels, explicit_starts = [], []
    for entry in state["signals"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("metadata"), Mapping):
            raise ValueError("Each signal needs samples and a metadata object")
        meta = dict(entry["metadata"])
        kind = meta.get("sensor_type")
        if not isinstance(kind, str) or kind.strip().lower() not in TYPES:
            raise ValueError(f"metadata.sensor_type must be one of {TYPES}")
        kind = kind.strip().lower()
        rate = positive_number(meta.get("sampling_rate_hz"), "metadata.sampling_rate_hz")
        if "powerline_hz" in meta:
            powerline = meta["powerline_hz"]
            if isinstance(powerline, (bool, np.bool_)) or not isinstance(powerline, Real) or powerline not in (0, 50, 60):
                raise ValueError("metadata.powerline_hz must be numeric 0, 50, or 60")
        raw = np.asarray(entry.get("samples"))
        if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not 2 <= raw.size <= max_samples:
            raise ValueError(f"samples must contain 2–{max_samples} real numeric measurements")
        values = raw.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Samples must be finite; missing intervals cannot be filled silently")
        values, unit = canonical_unit(values, kind, meta.get("unit"))
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Unit conversion overflowed the waveform representation")
        start = meta.get("start_time_s", 0.0)
        if isinstance(start, (bool, np.bool_)) or not isinstance(start, Real) or not math.isfinite(start):
            raise ValueError("metadata.start_time_s must be finite")
        if "duration_s" in meta:
            duration = positive_number(meta["duration_s"], "metadata.duration_s")
            if not math.isclose(duration, len(values)/rate, rel_tol=0, abs_tol=max(1e-9, 0.5/rate)):
                raise ValueError("metadata.duration_s disagrees with samples and sampling rate")
        explicit_starts.append("start_time_s" in meta)
        if "timestamps_s" in meta:
            ts = np.asarray(meta["timestamps_s"], dtype=np.float64)
            if ts.shape != values.shape or not np.isfinite(ts).all():
                raise ValueError("Timestamps must be finite and match samples")
            if not np.allclose(np.diff(ts), 1/rate, rtol=0.01, atol=1e-9):
                raise ValueError("Timestamps disagree with the declared uniform sampling rate")
            if not math.isclose(ts[0], start, rel_tol=0, abs_tol=max(1e-9, 0.01/rate)):
                raise ValueError("Timestamps disagree with start_time_s")
        for field in ("axis", "lead", "placement", "reference"):
            if field in meta and (not isinstance(meta[field], str) or not meta[field].strip()):
                raise ValueError(f"metadata.{field} must be a nonempty string when supplied")
        if meta.get("axis", "x").lower() not in ("x", "y", "z"):
            raise ValueError("Inertial axis must be x, y, or z")
        meta.update(sensor_type=kind, sampling_rate_hz=rate, unit=unit)
        channels.append(Channel(values, kind, rate, float(start), meta))
    if len({c.rate for c in channels}) > 1 and not all(explicit_starts):
        raise ValueError("Heterogeneous rates require start_time_s on every channel")
    first = channels[0]
    for c in channels[1:]:
        sample_interval = min(1/first.rate, 1/c.rate)
        if (abs(c.start-first.start) > max(1e-9, 0.01*sample_interval)
                or abs(len(c.samples)/c.rate-len(first.samples)/first.rate) > max(1e-9, 0.5*sample_interval)):
            raise ValueError("Channels must describe the same measured interval")
    # Canonical ordering makes channel permutations produce identical preparations.
    return sorted(channels, key=lambda c: (c.sensor_type, str(c.metadata.get("lead", "")),
                  str(c.metadata.get("placement", "")), str(c.metadata.get("axis", "")),
                  c.samples.tobytes()))
