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
    """Return broad and dynamic waveforms with native-time quality observations."""
    raw = channel.samples.astype(np.float64)
    rate, kind = channel.rate, channel.sensor_type
    audit = {"sensor_type": kind, "sampling_rate_hz": rate, "unit": channel.metadata["unit"],
             "observations": len(raw), "flatline": bool(np.ptp(raw) == 0),
             "minimum": float(raw.min()), "maximum": float(raw.max())}
    values = raw
    powerline = channel.metadata.get("powerline_hz", 0)
    if powerline and kind in ("ecg", "eeg", "emg"):
        if powerline+1 >= rate/2:
            raise UnsupportedSignalError("Declared powerline band is outside the native Nyquist limit")
        # NeuroKit selects a narrow band-stop when lowcut exceeds highcut.
        values = _filter(values, rate, low=powerline+1, high=powerline-1, order=2)
        audit["powerline_hz"] = powerline

    dynamic = None
    if kind == "ecg":
        broad = _filter(values, rate, high=min(150, 0.4*rate))
        if rate > 2:
            # This is NeuroKit's ECG high-pass stage, without its implicit 50 Hz smoothing.
            dynamic = _filter(values, rate, low=0.5, order=5)
            audit["cleaning"] = "broad: DC-preserving low-pass <=150 Hz; detection: NeuroKit 0.5 Hz high-pass"
            _observe(audit, "beats", lambda: _beats(audit, dynamic, rate, kind))
        else:
            audit["cleaning"] = "broad low-pass; native rate insufficient for ECG detection"

    elif kind == "ppg":
        broad = _filter(values, rate, high=min(40, 0.4*rate))
        if rate > 16:
            dynamic = nk.ppg_clean(values, sampling_rate=rate, method="elgendi")
            _observe(audit, "beats", lambda: _beats(audit, dynamic, rate, kind))
        audit["cleaning"] = "broad: DC-preserving low-pass <=40 Hz; detection: Elgendi 0.5-8 Hz when native rate permits"

    elif kind == "eda":
        broad = nk.eda_clean(values, sampling_rate=rate, method="neurokit" if rate > 6 else "none")
        audit["cleaning"] = "NeuroKit 3 Hz low-pass" if rate > 6 else "native <=6 Hz: no added low-pass"
        if rate > 0.125 and len(values) > 18 and not audit["flatline"]:
            components = nk.eda_phasic(broad, sampling_rate=rate, method="highpass", cutoff=0.05)
            dynamic = components["EDA_Phasic"].to_numpy()
            def responses():
                _, info = nk.eda_peaks(dynamic, sampling_rate=rate, method="neurokit", amplitude_min=0.1)
                return {"method": "highpass-0.05Hz/neurokit", **_events(info, ["SCR_Onsets", "SCR_Peaks"], rate)}
            _observe(audit, "responses", responses)

    elif kind == "respiration":
        broad = _filter(values, rate, high=min(3, 0.4*rate))
        audit["cleaning"] = "broad: DC-preserving low-pass <=3 Hz; detection: khodadad2018"
        if rate > 6:
            dynamic = nk.rsp_clean(values, sampling_rate=rate, method="khodadad2018")
            def breaths():
                _, info = nk.rsp_peaks(dynamic, sampling_rate=rate, method="khodadad2018")
                return {"method": "khodadad2018", **_events(info, ["RSP_Peaks", "RSP_Troughs"], rate)}
            _observe(audit, "breaths", breaths)

    elif kind == "emg":
        if rate <= 50:
            raise UnsupportedSignalError("Raw EMG needs a native rate above 50 Hz for the 20 Hz high-pass")
        broad = _filter(values, rate, low=20, high=min(450, 0.4*rate))
        envelope = np.maximum(0, _filter(np.abs(broad), rate, high=8, order=2))
        audit["cleaning"] = "20-min(450,0.4*fs) Hz band-pass; rectified 8 Hz envelope for activation checks only"
        def activations():
            _, info = nk.emg_activation(emg_amplitude=envelope, sampling_rate=rate, method="threshold")
            return {"method": "rectified-8Hz/threshold", **_events(info, ["EMG_Onsets", "EMG_Offsets"], rate)}
        _observe(audit, "activation", activations)

    elif kind == "eeg":
        broad = _filter(values, rate, high=min(100, 0.4*rate))
        audit["cleaning"] = "DC-preserving low-pass <=100 Hz; supplied reference retained"

    elif kind in ("accelerometer", "gyroscope"):
        broad = _filter(values, rate, high=min(40, 0.4*rate))
        audit["cleaning"] = "DC-preserving low-pass <=40 Hz; signed axes and gravity retained"

    else:  # temperature, already validated against the nine supported types
        broad = nk.signal_smooth(np.pad(values, 1, mode="edge"), kernel="median", size=3)[1:-1]
        audit["cleaning"] = "three-observation median; absolute level and trend retained"

    dynamic = broad if dynamic is None else dynamic
    for waveform in (broad, dynamic):
        if waveform.shape != raw.shape or not np.isfinite(waveform).all():
            raise UnsupportedSignalError(f"Invalid {kind} waveform after native-rate cleaning")
    return broad, dynamic, audit


def check_eeg_channels(channels, waveforms, audits):
    """Flag aligned EEG channel outliers without removing channels."""
    groups = {}
    for i, c in enumerate(channels):
        if c.sensor_type == "eeg":
            groups.setdefault((c.rate, len(c.samples), c.metadata.get("reference")), []).append(i)
    for indices in groups.values():
        if len(indices) < 4:
            continue
        report = {}
        _observe(report, "channels", lambda: {"bad_indices": nk.eeg_badchannels(
            np.stack([waveforms[i] for i in indices]))[0]})
        result = report["channels"]
        for local, i in enumerate(indices):
            audits[i]["channel_quality"] = {"status": result["status"], "method": "eeg_badchannels",
                                           "flagged": local in result.get("bad_indices", [])}
            if result["status"] == "unavailable":
                audits[i]["channel_quality"]["reason"] = result["reason"]
