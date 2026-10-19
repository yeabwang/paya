# Signal preparation

`validate → clean raw channels only → native variable-length formatting → frozen MantisV2`

## Input contract

Each physical channel contains a one-dimensional `samples` array and `metadata`:

| Metadata | Requirement |
| --- | --- |
| `sensor_type` | ECG, EEG, accelerometer, gyroscope, PPG, EDA, respiration, temperature or EMG; case-insensitive |
| `sampling_rate_hz` | Required finite positive number, never inferred or replaced |
| `duration_s` | Required finite positive number matching `sample_count / sampling_rate_hz` |
| `signal_condition` | `raw` by default, or `processed` |

Samples must be finite and representable in float32. Their supplied amplitude scale is retained; unit conversion and lead handling are absent. Channels must describe the same interval. Mixed native rates require matching explicit `start_time_s` values. Optional timestamps must match the declared uniform rate.

## Raw NeuroKit processing

[physiology.py](../paya/physiology.py) returns one native-rate cleaned waveform plus audit observations. These operations do not change sample count or sampling rate. `fs` below is the supplied rate.

| Signal | Retained waveform | Audit calculations |
| --- | --- | --- |
| ECG | DC-preserving low-pass `min(150,0.4*fs)` | High-pass detection copy; peaks and template quality |
| PPG | DC-preserving low-pass `min(40,0.4*fs)` | Elgendi detection copy where rate permits; peaks and template quality |
| EDA | NeuroKit 3 Hz cleaning if `fs>6`; otherwise no added low-pass | Phasic responses where coverage permits |
| Respiration | DC-preserving low-pass `min(3,0.4*fs)` | Khodadad2018 detection copy; peaks and troughs |
| EMG | Bandpass 20–`min(450,0.4*fs)`; requires `fs>50` | Rectified 8 Hz envelope and activation |
| EEG | DC-preserving low-pass `min(100,0.4*fs)` | Bad-channel flags for aligned raw EEG groups |
| Accelerometer/gyroscope | DC-preserving low-pass `min(40,0.4*fs)` | Range and flatline flags |
| Temperature | Three-observation median with edge context | Range and flatline flags |

Optional electrical `powerline_hz` accepts numeric `0`, `50` or `60`; no frequency is assumed. A declared notch band must fit below native Nyquist. Detection copies and quality measurements are audit-only. Detector failure preserves the cleaned waveform and records unavailability rather than synthesizing normal physiology.

## Processed signals

`signal_condition: processed` skips all NeuroKit cleaning, peak detection and quality calculations. Validation and native window alignment still run. Provided samples are unchanged apart from float32 representation. Mantis retains its own frozen normalization and patch-statistics operations.

## Native window alignment

Use the supplied recording as context, without a fixed physiological window duration. Let `L=32*floor(N/32)` for a channel with `N` samples. Encode the complete array once when aligned; otherwise encode the length-`L` prefix and suffix. This covers all real measurements without resampling, interpolation, amplitude normalization or external sample padding.

| Recording | Encoder windows | Elapsed intervals |
| --- | --- | --- |
| 10 s at 125 Hz, 1,250 samples | Two × 1,248 samples | `[0,9.984]`, `[0.016,10]` seconds |
| 6 s at 65 Hz, 390 samples | Two × 384 samples | `[0,384/65]`, `[6/65,6]` seconds |
| 8 s at 64 Hz, 512 samples | One × 512 samples | `[0,8]` seconds |

Numerical validity requires at least 64 native measurements: the frozen model computes sample standard deviation in 32 patches and needs two observations per patch. This is not a physiological minimum. Defaults cap each channel at 1,000,000 samples and each request at 64 channels/128 embeddings. Equal-length encoder batches are bounded by the configured sample budget. No recording is shortened to fit a budget; over-budget requests fail explicitly.

## Physiological scope

The recording duration remains available through elapsed start/end encoding. Normalized coordinates alone would discard physical time scale. All supplied context is represented, although the frozen encoder still compresses it into 32 patches.

Acceptance does not establish suitability for every question. Pulse shape, pulse rate, respiration and HRV require different evidence and durations. NeuroKit, for example, recommends minutes of high-quality data for several frequency-domain HRV measures. Short recordings are not extended to pretend that this information exists. Generalization and calibration need evaluation by task, duration, rate and domain.

References: [Mantis data format](https://github.com/vfeofanov/mantis#prepare-data), [NeuroKit custom processing](https://neuropsychology.github.io/NeuroKit/examples/bio_custom/bio_custom.html), [PPG](https://neuropsychology.github.io/NeuroKit/functions/ppg.html), [HRV duration requirements](https://neuropsychology.github.io/NeuroKit/functions/hrv.html#hrv-frequency).
