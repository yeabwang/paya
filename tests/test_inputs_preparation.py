import numpy as np
import pytest
import torch

from paya.inputs import TYPES, validate_state, validate_text, UnsupportedSignalError
from paya.preprocessing import SignalPreprocessor, PreparationConfig, native_windows
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


@pytest.mark.parametrize("duration", [None, "", "6", True, 0, -1, float("nan"), float("inf")])
def test_strict_required_duration(duration):
    request = state()
    request["signals"][0]["metadata"]["duration_s"] = duration
    with pytest.raises(ValueError, match="duration_s"):
        validate_state(request)


@pytest.mark.parametrize("field", ["sensor_type", "sampling_rate_hz", "duration_s"])
def test_required_metadata_cannot_be_omitted(field):
    request = state()
    del request["signals"][0]["metadata"][field]
    with pytest.raises(ValueError, match=field):
        validate_state(request)


@pytest.mark.parametrize("condition", [None, "", "clean", "RAW", True, 1, []])
def test_strict_signal_condition(condition):
    request = state()
    request["signals"][0]["metadata"]["signal_condition"] = condition
    with pytest.raises(ValueError, match="signal_condition"):
        validate_state(request)


@pytest.mark.parametrize("kind", TYPES)
def test_native_preparations_for_all_nine_signals(kind):
    rate = {"temperature": 1, "eda": 4, "ppg": 73, "accelerometer": 123,
            "gyroscope": 80, "respiration": 32, "emg": 250}.get(kind, 360)
    processor = SignalPreprocessor()
    prepared = processor.prepare(state(kind, rate, max(6, 64/rate)))
    channel = prepared.channels[0]
    assert channel.rate == rate and channel.metadata["signal_condition"] == "raw"
    frames = processor.format(prepared)
    assert frames.intervals_s.shape == (len(frames.frames), 2)
    for frame in frames.frames:
        assert frame.shape[0] == 1 and frame.shape[1] >= 64 and frame.shape[1] % 32 == 0
        assert frame.dtype == torch.float32 and torch.isfinite(frame).all()
    torch.testing.assert_close(frames.frames[0][0], torch.from_numpy(channel.samples[:frames.frames[0].shape[1]]), atol=0, rtol=0)
    torch.testing.assert_close(frames.intervals_s[-1, 1], torch.tensor(len(channel.samples)/rate))


def test_samples_keep_their_scale_without_unit_or_lead_fields():
    request = state("temperature", 1, 512, condition="processed")
    request["signals"][0]["samples"] = [310.0]*512
    channel = validate_state(request)[0]
    assert "unit" not in channel.metadata and "lead" not in channel.metadata
    np.testing.assert_array_equal(channel.samples, np.full(512, 310, dtype=np.float32))
    # Removed fields cannot trigger implicit amplitude conversion.
    request["signals"][0]["metadata"].update(unit="F", lead="II")
    other = validate_state(request)[0]
    np.testing.assert_array_equal(other.samples, channel.samples)
    assert "unit" not in other.metadata and "lead" not in other.metadata
    slow_rate = validate_state(state("temperature", 0.25, 2048))[0]
    assert slow_rate.rate == 0.25


@pytest.mark.parametrize("kind", TYPES)
def test_processed_channels_bypass_entire_neurokit_pipeline(kind, monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Processed data must bypass all NeuroKit processing and quality checks")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    monkeypatch.setattr(module, "check_eeg_channels", unexpected)
    processor = SignalPreprocessor()
    # Even processed EMG at a low rate must not enter the raw EMG filter.
    request = state(kind, 20, 30, condition="processed")
    supplied = np.asarray(request["signals"][0]["samples"], dtype=np.float32)
    prepared = processor.prepare(request)
    np.testing.assert_array_equal(prepared.channels[0].samples, supplied)
    assert prepared.audit["quality"][0]["cleaning"] == "skipped"
    frames = processor.format(prepared)
    assert [tuple(frame.shape) for frame in frames.frames] == [(1, 576), (1, 576)]
    np.testing.assert_array_equal(frames.frames[0][0], supplied[:576])
    np.testing.assert_array_equal(frames.frames[1][0], supplied[-576:])


def test_raw_is_default_and_mixed_conditions_clean_only_raw_channels(monkeypatch):
    import paya.preprocessing as module
    calls = []
    def clean(channel):
        calls.append(channel.sensor_type)
        return channel.samples+1, {"cleaning": "test"}
    monkeypatch.setattr(module, "clean_channel", clean)
    request = state()
    request["signals"].append(state("ppg", condition="processed")["signals"][0])
    prepared = SignalPreprocessor().prepare(request)
    assert calls == ["ecg"]
    np.testing.assert_array_equal(prepared.channels[0].samples,
                                  np.asarray(request["signals"][0]["samples"], dtype=np.float32)+1)
    np.testing.assert_array_equal(prepared.channels[1].samples,
                                  np.asarray(request["signals"][1]["samples"], dtype=np.float32))


@pytest.mark.parametrize("rate,duration,length,starts", [
    (125, 10, 1248, [0, 2]), (65, 6, 384, [0, 6]), (64, 8, 512, [0]),
    (1, 64, 64, [0]), (123.5, 10, 1216, [0, 19]),
])
def test_native_full_context_coverage_without_resampling_or_padding(rate, duration, length, starts):
    processor = SignalPreprocessor()
    request = state("ppg", rate, duration, condition="processed")
    prepared = processor.prepare(request)
    frames = processor.format(prepared)
    samples = prepared.channels[0].samples
    assert [tuple(frame.shape) for frame in frames.frames] == [(1, length)]*len(starts)
    coverage = np.zeros(len(samples), dtype=bool)
    for index, (start, frame) in enumerate(zip(starts, frames.frames)):
        np.testing.assert_array_equal(frame[0], samples[start:start+length])
        coverage[start:start+length] = True
        torch.testing.assert_close(frames.intervals_s[index], torch.tensor([start/rate, (start+length)/rate]))
    assert coverage.all()
    assert frames.intervals_s[0, 0] == 0 and frames.intervals_s[-1, 1] == duration


@pytest.mark.parametrize("length", [2, 31, 32, 33, 63])
def test_actual_encoder_minimum_is_checked_before_neurokit(length, monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Numerically unsupported input must fail before cleaning")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    with pytest.raises(UnsupportedSignalError, match="64 native samples"):
        SignalPreprocessor().prepare(state("ppg", 1, length))


def test_all_alignment_remainders_cover_the_original_recording():
    for count in range(64, 257):
        intervals = native_windows(count)
        coverage = np.zeros(count, dtype=bool)
        assert len(intervals) == (1 if count % 32 == 0 else 2)
        for start, end in intervals:
            assert end-start >= 64 and (end-start) % 32 == 0
            coverage[start:end] = True
        assert coverage.all()



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


def test_frame_budget_is_checked_before_neurokit(monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Over-budget requests must fail before cleaning")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    with pytest.raises(UnsupportedSignalError, match="budget"):
        SignalPreprocessor(PreparationConfig(max_signal_tokens=1)).prepare(state())
    with pytest.raises(ValueError):
        PreparationConfig(max_signal_tokens=0)


def test_sample_budget_must_allow_the_actual_encoder_minimum():
    with pytest.raises(ValueError, match="64 native samples"):
        PreparationConfig(max_samples=63)


@pytest.mark.parametrize("question,criteria", [("", ["a", "b"]), ("q", ["a"]),
    ("q", ["a", " a "]), ("q", ["a", ""]), ("q", {"a": "b", "c": "d"})])
def test_separate_question_and_ordered_criteria_validation(question, criteria):
    with pytest.raises(ValueError):
        validate_text(question, criteria)
