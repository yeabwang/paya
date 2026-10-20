"""Native-rate signal cleaning and physiological quality observations."""

import warnings

import neurokit2 as nk
import numpy as np

from .inputs import UnsupportedSignalError

def _filter(values, rate, low=None, high=None, order=4):
    return nk.signal_filter(values, sampling_rate=rate, lowcut=low, highcut=high,
                            method="butterworth", order=order)


def _events(info, keys, rate):
    result = {}
    for key in keys:
        indices = np.asarray(info[key], dtype=float)
        result[key] = (indices[np.isfinite(indices)]/rate).tolist()
    return result


def _cardiac_quality(cleaned, peaks, rate, kind):
    if len(peaks) < 4:
        raise ValueError("At least four detected beats are needed for template quality")
    if kind == "ecg":
        quality = nk.ecg_quality(cleaned, rpeaks=peaks, sampling_rate=rate, method="templatematch")
    else:
        quality = nk.ppg_quality(cleaned, peaks=peaks, sampling_rate=rate, method="templatematch")
    quality = np.asarray(quality, dtype=float)
    valid = quality[np.isfinite(quality)]
    if not len(valid):
        raise ValueError("No finite template quality values")
    return {"method": "templatematch", "median": float(np.median(valid)),
            "finite_fraction": float(len(valid)/len(quality))}


def _beats(audit, cleaned, rate, kind):
    detect, method, key = ((nk.ecg_peaks, "neurokit", "ECG_R_Peaks") if kind == "ecg"
                           else (nk.ppg_peaks, "elgendi", "PPG_Peaks"))
    _, info = detect(cleaned, sampling_rate=rate, method=method, correct_artifacts=False)
    detected = np.asarray(info[key], dtype=int)
    # Exclude incomplete boundary beats from measurements, retaining their times.
    interior = (detected >= 0.25*rate) & (detected < len(cleaned)-0.25*rate)
    peaks = detected[interior]
    _observe(audit, "template_quality", lambda: _cardiac_quality(cleaned, peaks, rate, kind))
    return {"method": method, **_events({key: peaks}, [key], rate),
            "boundary_peaks_s": (detected[~interior]/rate).tolist(), "boundary_margin_s": 0.25,
            "correct_artifacts": False}


def _observe(audit, name, operation):
    # Detector failure can reflect abnormal physiology; preserve the waveform.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            audit[name] = {"status": "available", **operation()}
        except (ValueError, IndexError, ZeroDivisionError) as error:
            audit[name] = {"status": "unavailable", "reason": str(error)}
    if caught:
        audit[name]["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))


def clean_channel(channel):
    """Return one cleaned waveform and native-time physiological observations."""
    raw = channel.samples.astype(np.float64)
    rate, kind = channel.rate, channel.sensor_type
    audit = {"sensor_type": kind, "sampling_rate_hz": rate,
             "observations": len(raw), "flatline": bool(np.ptp(raw) == 0),
             "minimum": float(raw.min()), "maximum": float(raw.max())}
    values = raw
    powerline = channel.metadata.get("powerline_hz", 0)
    if powerline and kind in ("ecg", "eeg", "ieeg"):
        if powerline+1 >= rate/2:
            raise UnsupportedSignalError("Declared powerline band is outside the native Nyquist limit")
        # NeuroKit selects a narrow band-stop when lowcut exceeds highcut.
        values = _filter(values, rate, low=powerline+1, high=powerline-1, order=2)
        audit["powerline_hz"] = powerline

    detection = None
    if kind == "ecg":
        broad = _filter(values, rate, high=min(150, 0.4*rate))
        if rate > 2:
            # This is NeuroKit's ECG high-pass stage, without its implicit 50 Hz smoothing.
            detection = _filter(values, rate, low=0.5, order=5)
            audit["cleaning"] = "broad: DC-preserving low-pass <=150 Hz; detection: NeuroKit 0.5 Hz high-pass"
            _observe(audit, "beats", lambda: _beats(audit, detection, rate, kind))
        else:
            audit["cleaning"] = "broad low-pass; native rate insufficient for ECG detection"

    elif kind == "ppg":
        broad = _filter(values, rate, high=min(40, 0.4*rate))
        if rate > 16:
            detection = nk.ppg_clean(values, sampling_rate=rate, method="elgendi")
            _observe(audit, "beats", lambda: _beats(audit, detection, rate, kind))
        audit["cleaning"] = "broad: DC-preserving low-pass <=40 Hz; detection: Elgendi 0.5-8 Hz when native rate permits"

    elif kind == "eeg":
        broad = _filter(values, rate, high=min(100, 0.4*rate))
        audit["cleaning"] = "DC-preserving low-pass <=100 Hz; supplied reference retained"

    elif kind == "ieeg":
        broad = _filter(values, rate, high=0.4*rate)
        audit["cleaning"] = "DC-preserving low-pass at 0.4*fs; supplied intracranial reference retained"

    else:
        raise UnsupportedSignalError(f"Unsupported sensor type: {kind}")

    if broad.shape != raw.shape or not np.isfinite(broad).all():
        raise UnsupportedSignalError(f"Invalid {kind} waveform after native-rate cleaning")
    return broad, audit


def check_eeg_channels(channels, waveforms, audits):
    """Flag aligned EEG channel outliers without removing channels."""
    indices = [i for i, c in enumerate(channels) if c.sensor_type == "eeg"]
    if len(indices) < 4 or len({channels[i].metadata.get("reference") for i in indices}) != 1:
        return
    report = {}
    _observe(report, "channels", lambda: {"bad_indices": nk.eeg_badchannels(
        np.stack([waveforms[i] for i in indices]))[0]})
    result = report["channels"]
    for local, i in enumerate(indices):
        audits[i]["channel_quality"] = {"status": result["status"], "method": "eeg_badchannels",
                                       "flagged": local in result.get("bad_indices", [])}
        if result["status"] == "unavailable":
            audits[i]["channel_quality"]["reason"] = result["reason"]
