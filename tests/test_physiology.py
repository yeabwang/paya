"""Known timing, levels and frequencies check physiological preparation, not QA accuracy."""

import json

import neurokit2 as nk
import numpy as np
import pytest

from paya.inputs import validate_state
from paya.physiology import clean_channel
from paya.preprocessing import SignalPreprocessor
from conftest import state


def request(kind, values, rate, **metadata):
    item = state(kind, rate, len(values)/rate)
    item["signals"][0]["samples"] = np.asarray(values).tolist()
    item["signals"][0]["metadata"].update(metadata)
    return item


def ecg(duration=20, rate=500):
    t = np.arange(int(duration*rate))/rate
    beats = np.arange(1, duration-1, 0.8)
    values = np.full(len(t), 2.0)
    for beat in beats:
        for offset, width, amplitude in [(-0.18, 0.035, 0.12), (-0.03, 0.01, -0.15),
                                         (0, 0.012, 1), (0.03, 0.01, -0.25), (0.24, 0.06, 0.3)]:
            values += amplitude*np.exp(-0.5*((t-beat-offset)/width)**2)
    return values, beats


def amplitude(values, frequency, rate):
    # Crop filter-boundary context before measuring a known sinusoidal component.
    values = values[int(rate):-int(rate)]
    t = np.arange(len(values))/rate
    return abs(2*np.mean((values-values.mean())*np.exp(-2j*np.pi*frequency*t)))


def test_ecg_native_beat_timing_and_morphology_are_retained():
    values, expected = ecg()
    t = np.arange(len(values))/500
    values += 0.1*np.sin(2*np.pi*50*t)
    channel = validate_state(request("ecg", values, 500, powerline_hz=50))[0]
    broad, audit = clean_channel(channel)
    actual = np.asarray(audit["beats"]["ECG_R_Peaks"])
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, atol=0.012, rtol=0)
    assert audit["beats"]["correct_artifacts"] is False
    assert audit["template_quality"]["status"] == "available"
    assert abs(broad.mean()-values.mean()) < 0.005
    assert amplitude(broad, 50, 500) < amplitude(values, 50, 500)*0.1
    assert np.max(broad)-np.median(broad) > 0.9
    json.dumps(audit, allow_nan=False)


def test_ppg_cleaning_keeps_intensity_and_finds_native_pulses():
    rate = 64
    t = np.arange(20*rate)/rate
    phase = (t % 0.8)/0.8
    pulse = phase**2*np.exp(-phase*10)
    values = 5 + pulse/pulse.max()
    broad, audit = clean_channel(validate_state(request("ppg", values, rate))[0])
    assert abs(broad.mean()-values.mean()) < 0.01
    peaks = np.asarray(audit["beats"]["PPG_Peaks"])
    assert len(peaks) >= 20
    np.testing.assert_allclose(np.diff(peaks), 0.8, atol=2/rate, rtol=0)
    assert audit["template_quality"]["status"] == "available"


def test_ieeg_native_bandwidth_and_declared_notch_are_retained():
    rate = 1000
    t = np.arange(8*rate)/rate
    values = 2+0.5*np.sin(2*np.pi*160*t)+0.1*np.sin(2*np.pi*60*t)
    broad, audit = clean_channel(validate_state(request("ieeg", values, rate, powerline_hz=60))[0])
    assert broad.shape == values.shape and np.isfinite(broad).all()
    assert abs(broad.mean()-2) < 0.005
    assert amplitude(broad, 160, rate) > 0.48
    assert amplitude(broad, 60, rate) < 0.01
    assert audit["powerline_hz"] == 60 and "beats" not in audit


def test_scalp_eeg_channel_quality_is_not_applied_to_ieeg(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Scalp EEG outlier checks must not be applied to iEEG")
    monkeypatch.setattr(nk, "eeg_badchannels", unexpected)
    channel = state("ieeg", 500, 6)["signals"][0]
    prepared = SignalPreprocessor().prepare({"signals": [channel]*4})
    assert len(prepared.channels) == 4
    assert all("channel_quality" not in audit for audit in prepared.audit["quality"])


def test_failed_detector_records_unavailability_without_destroying_waveform(monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("No reliable beats")
    monkeypatch.setattr(nk, "ecg_peaks", failed)
    cleaned = SignalPreprocessor().prepare(request("ecg", ecg(6)[0], 500))
    assert cleaned.audit["quality"][0]["beats"]["status"] == "unavailable"
    assert all(np.isfinite(channel.samples).all() for channel in cleaned.channels)


def test_quality_records_are_audit_only_and_old_preparations_are_rejected(model):
    prepared = model.preprocessor.prepare(request("ecg", ecg(6)[0], 500))
    before = model.predict(prepared, "Which category?", ["a", "b"])
    prepared.audit["quality"] = [{"template_quality": {"median": -1}, "dataset_id": "unrelated"}]
    assert model.predict(prepared, "Which category?", ["a", "b"]) == before
    prepared.audit["preparation"] = {"contract": "unrecognized"}
    with pytest.raises(ValueError, match="current preprocessing"):
        model.predict(prepared, "Which category?", ["a", "b"])


def test_eeg_quality_flags_do_not_remove_or_rereference_channels():
    rate = 128
    t = np.arange(16*rate)/rate
    channels = []
    for i in range(6):
        values = 20*np.sin(2*np.pi*(8+i)*t)+(50 if i == 5 else i)
        channels.append(request("eeg", values, rate, placement=f"channel-{i}", reference="linked-mastoids")["signals"][0])
    processor = SignalPreprocessor()
    prepared = processor.prepare({"signals": channels})
    assert len(prepared.audit["quality"]) == 6
    for audit in prepared.audit["quality"]:
        assert audit["channel_quality"]["status"] == "available"
    samples = processor.format(prepared).samples
    assert samples.shape == (6, 2048)
    assert len(prepared.channels) == 6
    json.dumps(prepared.audit, allow_nan=False)


@pytest.mark.parametrize("frequency", ["50", True, None, 45, float("nan")])
def test_powerline_metadata_is_explicit_and_validated(frequency):
    with pytest.raises(ValueError, match="powerline_hz"):
        validate_state(request("ecg", ecg(6)[0], 500, powerline_hz=frequency))
