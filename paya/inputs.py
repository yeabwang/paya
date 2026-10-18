"""Validate physical signal requests and encode canonical sensor metadata."""

import math

import numpy as np
import torch

ENDING = "The answer is:"
SCHEMA_VERSION = 1
TYPES = (
    "unknown",
    "ecg",
    "eeg",
    "accelerometer",
    "gyroscope",
    "ppg",
    "eda",
    "respiration",
    "temperature",
    "emg",
)
AXES = ("unknown", "x", "y", "z")
PLACEMENTS = ("unknown", "chest", "wrist", "head", "waist", "ankle")
LEADS = ("unknown", "mlii", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6")
UNITS = ("unknown", "mv", "uv", "g", "rad/s", "us", "c", "standardized", "normalized")
METADATA_DIM = sum(map(len, (TYPES, AXES, PLACEMENTS, LEADS, UNITS))) + 2
# Unit conversion preserves information; no source-selected normalization/resampling.
CONVERSIONS = {
    "v": ("mv", 1000.0),
    "mv": ("mv", 1.0),
    "uv": ("mv", 0.001),
    "g": ("g", 1.0),
    "m/s2": ("g", 1 / 9.80665),
    "rad/s": ("rad/s", 1.0),
    "deg/s": ("rad/s", math.pi / 180),
}


def physical_metadata(channel, rate, length):
    """Encode documented sensor fields, log sample rate, and log duration.

    Return a float32 vector of length METADATA_DIM. Unrecognized categories use
    the explicit unknown slot; category ordering is part of the model contract.
    """
    fields = (
        ("sensor_type", TYPES),
        ("axis", AXES),
        ("placement", PLACEMENTS),
        ("lead", LEADS),
        ("unit", UNITS),
    )
    values = []
    for name, vocabulary in fields:
        value = str(channel.get(name, "unknown")).lower()
        value = value if value in vocabulary else "unknown"
        values.extend(float(value == token) for token in vocabulary)
    values.extend((math.log(rate / 65), math.log(length / rate)))
    return torch.tensor(values, dtype=torch.float32)


def prepare_signal(signal):
    """Validate a signal mapping and return waveform and metadata tensors.

    The mapping contains sampling_rate_hz and channels with one-dimensional
    samples plus optional physical descriptors. Channels must share length and
    rate. Known units are converted without resampling or normalization.

    Returns float32 tensors shaped [channels, samples] and
    [channels, METADATA_DIM]. Raises ValueError for unsupported dimensions,
    nonfinite values, invalid rates, or inconsistent channel lengths.
    """
    rate = float(signal["sampling_rate_hz"])
    if not math.isfinite(rate) or not 1 <= rate <= 4000:
        raise ValueError("sampling_rate_hz must be finite and between 1 and 4000")
    channels = signal.get("channels", [])
    if not channels or len(channels) > 64:
        raise ValueError("Supply 1Ã¢â‚¬â€œ64 physical channels")
    arrays, metadata, length = [], [], None
    for channel in channels:
        x = np.asarray(channel["samples"], dtype=np.float32)
        if x.ndim != 1 or not 10 <= len(x) <= 120000:
            raise ValueError(
                "Each channel needs a one-dimensional window of 10Ã¢â‚¬â€œ120000 samples"
            )
        if not np.isfinite(x).all():
            raise ValueError(
                "Signal measurements must be finite; missing channels must be omitted"
            )
        if length is not None and len(x) != length:
            raise ValueError(
                "Channels must share rate and length; align physical streams before inference"
            )
        length = len(x)
        unit = (
            str(channel.get("unit", "unknown"))
            .lower()
            .replace("Ã‚Âµ", "u")
            .replace("ÃŽÂ¼", "u")
        )
        normalized = {**channel, "unit": unit}
        if unit in CONVERSIONS:
            normalized["unit"], scale = CONVERSIONS[unit]
            x = x * scale
        if not np.isfinite(x).all():
            raise ValueError("Unit conversion produced nonfinite values")
        arrays.append(torch.from_numpy(x.copy()))
        metadata.append(physical_metadata(normalized, rate, length))
    return torch.stack(arrays), torch.stack(metadata)


def validate_question(question, choices):
    """Require a nonempty question and 2–64 distinct, nonempty choice descriptions."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Supply a nonempty question")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 64:
        raise ValueError("Supply 2Ã¢â‚¬â€œ64 choice descriptions")
    if any(not isinstance(c, str) or not c.strip() for c in choices) or len(
        set(choices)
    ) != len(choices):
        raise ValueError("Choice descriptions must be nonempty and distinct")
