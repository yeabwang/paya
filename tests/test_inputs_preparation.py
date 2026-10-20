import numpy as np
import pytest
import torch

from paya.inputs import TYPES, validate_state, validate_text
from paya.preprocessing import SignalPreprocessor, PreparationConfig
from conftest import state


@pytest.mark.parametrize("kind", [None, "", "  ", "unknown", "ekg", "typo", 1,
    "accelerometer", "gyroscope", "eda", "respiration", "temperature", "emg"])
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
def test_native_preparations_for_supported_signals(kind):
    rate = 73 if kind == "ppg" else 360
    processor = SignalPreprocessor()
    prepared = processor.prepare(state(kind, rate, 16))
    channel = prepared.channels[0]
    assert channel.rate == rate and channel.metadata["signal_condition"] == "raw"
    encoded_input = processor.format(prepared)
    assert encoded_input.samples.shape == (1, len(channel.samples))
    assert encoded_input.samples.dtype == torch.float32 and torch.isfinite(encoded_input.samples).all()
    torch.testing.assert_close(encoded_input.samples[0], torch.from_numpy(channel.samples), atol=0, rtol=0)


def test_public_sensor_types_and_ieeg_case_normalization():
    assert set(TYPES) == {"eeg", "ieeg", "ecg", "ppg"}
    assert validate_state(state("iEEG"))[0].sensor_type == "ieeg"


def test_samples_keep_their_scale_without_unit_or_lead_fields():
    request = state("ppg", 1, 512, condition="processed")
    request["signals"][0]["samples"] = [310.0]*512
    channel = validate_state(request)[0]
    assert "unit" not in channel.metadata and "lead" not in channel.metadata
    np.testing.assert_array_equal(channel.samples, np.full(512, 310, dtype=np.float32))
    # Removed fields cannot trigger implicit amplitude conversion.
    request["signals"][0]["metadata"].update(unit="F", lead="II")
    other = validate_state(request)[0]
    np.testing.assert_array_equal(other.samples, channel.samples)
    assert "unit" not in other.metadata and "lead" not in other.metadata
    slow_rate = validate_state(state("ppg", 0.25, 2048))[0]
    assert slow_rate.rate == 0.25


@pytest.mark.parametrize("kind", TYPES)
def test_processed_channels_bypass_entire_neurokit_pipeline(kind, monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Processed data must bypass all NeuroKit processing and quality checks")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    monkeypatch.setattr(module, "check_eeg_channels", unexpected)
    processor = SignalPreprocessor()
    request = state(kind, 20, 30, condition="processed")
    supplied = np.asarray(request["signals"][0]["samples"], dtype=np.float32)
    prepared = processor.prepare(request)
    np.testing.assert_array_equal(prepared.channels[0].samples, supplied)
    assert prepared.audit["quality"][0]["cleaning"] == "skipped"
    encoded_input = processor.format(prepared)
    assert encoded_input.samples.shape == (1, 600)
    np.testing.assert_array_equal(encoded_input.samples[0], supplied)


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


@pytest.mark.parametrize("rate,count", [
    (125, 1250), (65, 390), (1, 2), (123.5, 1235), (200, 12800),
])
def test_complete_recording_without_splitting_resampling_or_external_padding(rate, count):
    processor = SignalPreprocessor()
    request = state("ppg", rate, count/rate, condition="processed")
    prepared = processor.prepare(request)
    encoded_input = processor.format(prepared)
    samples = prepared.channels[0].samples
    assert encoded_input.samples.shape == (1, count)
    np.testing.assert_array_equal(encoded_input.samples[0], samples)


@pytest.mark.parametrize("length", [2, 31, 32, 33, 63])
def test_processed_short_signals_follow_encoder_normalization_minimum(length, monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Processed inputs must skip NeuroKit")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    prepared = SignalPreprocessor().prepare(state("ppg", 1, length, condition="processed"))
    assert SignalPreprocessor().format(prepared).samples.shape == (1, length)


def test_aligned_channels_are_stacked_in_supplied_order():
    request = state("ppg", 125, 10, condition="processed")
    request["signals"].append(state("ecg", 125, 10, condition="processed")["signals"][0])
    request["signals"][0]["samples"] = [2.0]*1250
    request["signals"][1]["samples"] = [3.0]*1250
    prepared = SignalPreprocessor().prepare(request)
    encoded_input = SignalPreprocessor().format(prepared)
    assert [c.sensor_type for c in prepared.channels] == ["ppg", "ecg"]
    assert encoded_input.samples.shape == (2, 1250)
    torch.testing.assert_close(encoded_input.samples[:, 0], torch.tensor([2., 3.]))


def test_different_native_rates_are_rejected_before_neurokit(monkeypatch):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Incompatible channels must fail before cleaning")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    request = state("ecg", 200, 10, condition="processed")
    request["signals"].append(state("ppg", 125, 10, condition="processed")["signals"][0])
    for channel in request["signals"]:
        channel["metadata"]["start_time_s"] = 0
    with pytest.raises(ValueError, match="same sampling rate and sample count"):
        SignalPreprocessor().prepare(request)



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


def test_channel_start_times_must_match():
    request = state()
    request["signals"].append(state("ppg")["signals"][0])
    assert len(validate_state(request)) == 2
    request["signals"][1]["metadata"]["start_time_s"] = 1
    with pytest.raises(ValueError, match="same start_time_s"):
        validate_state(request)


def test_channel_sample_counts_must_match():
    request = state("eeg", 500, 6, condition="processed")
    request["signals"].append(state("ieeg", 500, 5, condition="processed")["signals"][0])
    with pytest.raises(ValueError, match="same sampling rate and sample count"):
        validate_state(request)


def test_duration_and_timestamp_consistency():
    request = state()
    request["signals"][0]["metadata"]["duration_s"] = 20
    with pytest.raises(ValueError, match="duration_s"):
        validate_state(request)
    request = state()
    request["signals"][0]["metadata"]["timestamps_s"] = (np.arange(3000)/500+0.001).tolist()
    with pytest.raises(ValueError, match="start_time_s"):
        validate_state(request)


@pytest.mark.parametrize("condition", ["raw", "processed"])
def test_overlength_recordings_are_rejected_before_neurokit(monkeypatch, condition):
    import paya.preprocessing as module
    def unexpected(*args, **kwargs):
        pytest.fail("Over-budget requests must fail before cleaning")
    monkeypatch.setattr(module, "clean_channel", unexpected)
    with pytest.raises(ValueError, match="12800"):
        SignalPreprocessor().prepare(state("ecg", 500, 12801/500, condition=condition))


def test_budgets_respect_encoder_capacity():
    with pytest.raises(ValueError, match="128 channels"):
        PreparationConfig(max_channels=129)
    with pytest.raises(ValueError, match="between 2 and 12800"):
        PreparationConfig(max_samples=1)
    with pytest.raises(ValueError, match="between 2 and 12800"):
        PreparationConfig(max_samples=12801)


@pytest.mark.parametrize("question,criteria", [("", ["a", "b"]), ("q", ["a"]),
    ("q", ["a", " a "]), ("q", ["a", ""]), ("q", {"a": "b", "c": "d"})])
def test_separate_question_and_ordered_criteria_validation(question, criteria):
    with pytest.raises(ValueError):
        validate_text(question, criteria)
