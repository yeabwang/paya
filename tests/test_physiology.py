"""Known timing, levels and frequencies check physiological preparation, not QA accuracy."""

import json

import neurokit2 as nk
import numpy as np
import pytest
import torch

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


def test_low_rate_eda_keeps_tonic_level_and_detects_responses():
    rate = 4
    t = np.arange(128*rate)/rate
    values = 5 + 0.005*t
    for onset in (10, 35):
        elapsed = np.maximum(0, t-onset)
        values += np.exp(-elapsed/3)-np.exp(-elapsed/0.5)
    broad, audit = clean_channel(validate_state(request("eda", values, rate))[0])
    np.testing.assert_allclose(broad, values, atol=5e-7, rtol=0)
    peaks = audit["responses"]["SCR_Peaks"]
    assert any(10 < peak < 15 for peak in peaks) and any(35 < peak < 40 for peak in peaks)
    processor = SignalPreprocessor()
    cleaned = processor.prepare(request("eda", values, rate))
    assert torch.stack(processor.format(cleaned).frames).mean() > 5
    json.dumps(cleaned.audit, allow_nan=False)


def test_respiration_preserves_baseline_and_tracks_complete_cycles():
    rate = 32
    t = np.arange(60*rate)/rate
    values = 3+0.4*np.sin(2*np.pi*0.2*t)+0.1*np.sin(2*np.pi*8*t)
    broad, audit = clean_channel(validate_state(request("respiration", values, rate))[0])
    assert abs(broad.mean()-3) < 0.01
    assert amplitude(broad, 0.2, rate) > 0.38
    assert amplitude(broad, 8, rate) < 0.005
    peaks = np.asarray(audit["breaths"]["RSP_Peaks"])
    assert len(peaks) >= 10
    np.testing.assert_allclose(np.diff(peaks)[1:-1], 5, atol=2/rate, rtol=0)


def test_emg_envelope_checks_use_the_actual_native_rate():
    onsets = []
    for rate in (500, 1000, 2000):
        t = np.arange(8*rate)/rate
        active = ((t >= 2) & (t < 3)) | ((t >= 5) & (t < 6))
        values = (0.02+active*2)*np.sin(2*np.pi*80*t)
        broad, audit = clean_channel(validate_state(request("emg", values, rate))[0])
        assert np.std(broad[(t > 2.2) & (t < 2.8)]) > 1.3  # retain the 80 Hz carrier
        found = np.asarray(audit["activation"]["EMG_Onsets"])
        assert len(found) == 2
        np.testing.assert_allclose(found, [2, 5], atol=0.1, rtol=0)
        onsets.append(found)
    np.testing.assert_allclose(onsets[0], onsets[-1], atol=0.02, rtol=0)


def test_temperature_level_trend_and_isolated_spike():
    processor = SignalPreprocessor()
    constant = processor.prepare(request("temperature", [37.0]*512, 1))
    torch.testing.assert_close(torch.stack(processor.format(constant).frames), torch.full((1, 1, 512), 37.0), atol=0, rtol=0)
    t = np.arange(32)
    values = 36.5+0.01*t
    values[15] += 5
    broad, audit = clean_channel(validate_state(request("temperature", values, 1))[0])
    np.testing.assert_allclose(broad[2:-2], (36.5+0.01*t)[2:-2], atol=0.011, rtol=0)
    assert audit["maximum"] > 40 and broad.max() < 37


@pytest.mark.parametrize("kind,unit,level", [("accelerometer", "m/s2", 9.80665), ("gyroscope", "rad/s", -0.4)])
def test_inertial_cleaning_preserves_signed_dc_and_motion(kind, unit, level):
    rate = 100
    t = np.arange(10*rate)/rate
    values = level+0.3*np.sin(2*np.pi*3*t)
    broad, _ = clean_channel(validate_state(request(kind, values, rate, axis="z"))[0])
    assert abs(broad.mean()-level) < 1e-4
    assert amplitude(broad, 3, rate) > 0.29


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
    frames = processor.format(prepared).frames
    assert len(frames) == 1 and frames[0].shape == (6, 2048)
    assert len(prepared.channels) == 6
    json.dumps(prepared.audit, allow_nan=False)


def test_five_minute_ecg_is_fully_covered_without_time_compression():
    processor = SignalPreprocessor()
    prepared = processor.prepare(request("ecg", ecg(300)[0], 500))
    encoded_input = processor.format(prepared)
    assert [tuple(frame.shape) for frame in encoded_input.frames] == [(1, 12800)]*12
    assert encoded_input.intervals_s[0, 0] == 0 and encoded_input.intervals_s[-1, 1] == 300
    coverage = np.zeros(len(prepared.channels[0].samples), dtype=bool)
    for frame, interval in zip(encoded_input.frames, encoded_input.intervals_s):
        start = round(float(interval[0])*500)
        np.testing.assert_array_equal(frame[0], prepared.channels[0].samples[start:start+12800])
        coverage[start:start+12800] = True
    assert coverage.all()


@pytest.mark.parametrize("frequency", ["50", True, None, 45, float("nan")])
def test_powerline_metadata_is_explicit_and_validated(frequency):
    with pytest.raises(ValueError, match="powerline_hz"):
        validate_state(request("ecg", ecg(6)[0], 500, powerline_hz=frequency))
