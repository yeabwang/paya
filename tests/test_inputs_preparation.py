import copy

import numpy as np
import pytest
import torch

from paya.inputs import TYPES, validate_state, validate_text, UnsupportedSignalError
from paya.preprocessing import SignalPreprocessor, PreparationConfig
from conftest import state


@pytest.mark.parametrize("kind", [None, "", "  ", "unknown", "ekg", "typo", 1])
def test_strict_sensor_types(kind):
    request = state()
    request["signals"][0]["metadata"]["sensor_type"] = kind
    with pytest.raises(ValueError, match="sensor_type"):
        validate_state(request)


@pytest.mark.parametrize("rate", [None, "", "500", "unknown", True, False, 0, -1, float("nan"), float("inf")])
def test_strict_rates(rate):
    request = state()
    request["signals"][0]["metadata"]["sampling_rate_hz"] = rate
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        validate_state(request)


@pytest.mark.parametrize("kind", TYPES)
def test_nine_preparations_and_shapes(kind):
    rate = {"temperature": 1, "eda": 4, "ppg": 64, "accelerometer": 100,
            "gyroscope": 100, "respiration": 32, "emg": 2000}.get(kind, 500)
    processor = SignalPreprocessor()
    cleaned = processor.clean(state(kind, rate, 16))
    assert cleaned.channels[0].samples.shape == (rate*16,)
    assert cleaned.channels[0].rate == rate and cleaned.compatible[1]
    for expert, length in enumerate((390, 512)):
        if cleaned.compatible[expert]:
            encoded_input = processor.format_for_encoder(cleaned, expert)
            assert encoded_input.frames.ndim == 3 and encoded_input.frames.shape[1:] == (1, length)
            assert encoded_input.bounds.shape == (len(encoded_input.frames), 2)
            assert encoded_input.frames.dtype == torch.float32 and torch.isfinite(encoded_input.frames).all()
    if kind in ("temperature", "emg"):
        assert not cleaned.compatible[0]


def test_units_case_and_native_rates():
    request = state()
    equivalent = copy.deepcopy(request)
    equivalent["signals"][0]["metadata"].update(sensor_type=" ECG ", unit="uV")
    equivalent["signals"][0]["samples"] = (np.asarray(request["signals"][0]["samples"])*1000).tolist()
    a, b = SignalPreprocessor().clean(request), SignalPreprocessor().clean(equivalent)
    for left, right in zip(a.channels, b.channels):
        np.testing.assert_allclose(left.samples, right.samples, atol=2e-6, rtol=2e-6)
    slow_rate = validate_state(state("temperature", 0.25, 64))[0]
    assert slow_rate.rate == 0.25


def test_level_preservation_and_no_fake_padding():
    request = state("temperature", 1, 16)
    request["signals"][0]["samples"] = [37.0]*16
    processor = SignalPreprocessor()
    cleaned = processor.clean(request)
    encoder_input = processor.format_for_encoder(cleaned, 1)
    torch.testing.assert_close(encoder_input.frames.mean(), torch.tensor(37.0), atol=0.02, rtol=0)
    with pytest.raises(UnsupportedSignalError):
        processor.clean(state("temperature", 1, 2))


def test_missing_rate_cannot_use_top_level_fallback():
    request = state()
    del request["signals"][0]["metadata"]["sampling_rate_hz"]
    request["sampling_rate_hz"] = 500
    with pytest.raises(ValueError):
        validate_state(request)


@pytest.mark.parametrize("samples", [[], [1], [1, float("nan")], [True, False], ["1", "2"], [[1, 2]]])
def test_invalid_samples(samples):
    request = state()
    request["signals"][0]["samples"] = samples
    with pytest.raises(ValueError):
        validate_state(request)


def test_mixed_rate_alignment_and_timestamps():
    request = state()
    request["signals"].append(state("ppg", 125)["signals"][0])
    with pytest.raises(ValueError, match="start_time_s"):
        validate_state(request)
    for channel in request["signals"]:
        channel["metadata"]["start_time_s"] = 0
    assert len(validate_state(request)) == 2
    request["signals"][1]["metadata"]["start_time_s"] = 1
    with pytest.raises(ValueError, match="same measured interval"):
        validate_state(request)


def test_physical_interval_checks_do_not_use_the_slowest_channel_tolerance():
    request = state("ecg", 500, 16)
    request["signals"].append(state("temperature", 1, 16)["signals"][0])
    for channel in request["signals"]:
        channel["metadata"]["start_time_s"] = 0
    request["signals"][1]["metadata"]["start_time_s"] = 0.5
    with pytest.raises(ValueError, match="same measured interval"):
        validate_state(request)
    request = state()
    request["signals"][0]["metadata"]["duration_s"] = 20
    with pytest.raises(ValueError, match="duration_s"):
        validate_state(request)
    request = state()
    request["signals"][0]["metadata"]["timestamps_s"] = (np.arange(3000)/500+0.001).tolist()
    with pytest.raises(ValueError, match="start_time_s"):
        validate_state(request)


def test_budget_checked_before_large_resampling():
    request = state()
    request["signals"][0]["metadata"]["sampling_rate_hz"] = 1e-100
    with pytest.raises(UnsupportedSignalError, match="budget"):
        SignalPreprocessor().clean(request)
    with pytest.raises(ValueError):
        PreparationConfig(time_bins=44)


@pytest.mark.parametrize("question,criteria", [("", ["a", "b"]), ("q", ["a"]),
    ("q", ["a", " a "]), ("q", ["a", ""]), ("q", {"a": "b", "c": "d"})])
def test_separate_question_and_ordered_criteria_validation(question, criteria):
    with pytest.raises(ValueError):
        validate_text(question, criteria)
