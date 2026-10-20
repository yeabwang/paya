# Signal preparation

`validate → clean raw channels only → group aligned native channels → frozen SPOTR`

## Input contract

Each physical channel contains a one-dimensional `samples` array and `metadata`:

| Metadata | Requirement |
| --- | --- |
| `sensor_type` | ECG, EEG, accelerometer, gyroscope, PPG, EDA, respiration, temperature or EMG; case-insensitive |
| `sampling_rate_hz` | Required finite positive number, never inferred or replaced |
| `duration_s` | Required finite positive number matching `sample_count / sampling_rate_hz` |
| `signal_condition` | `raw` by default, or `processed` |

Samples must be finite and representable in float32. Preprocessing retains their supplied amplitude scale; unit conversion and lead handling are absent. Channels must describe the same interval. Mixed native rates require matching explicit `start_time_s` values. Optional timestamps must match the declared uniform rate. Supplied channel order is preserved for SPOTR's learned channel positions.

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

`signal_condition: processed` skips all NeuroKit cleaning, peak detection and quality calculations. Validation, grouping and window formation still run. Provided samples enter SPOTR unchanged apart from float32 representation; its own pretrained normalization and internal padding apply to both raw and processed inputs.

## Native grouping and windows

Aligned channels with the same rate and sample count form one `[C,L]` group. Different rates remain separate groups. Encode each group once when `L<=12800`. Longer groups use complete 12,800-sample windows, with a final overlapping window ending at the last measurement when necessary. This covers every sample without resampling or external padding. The limit follows the checkpoint's 128 temporal positions and 100-sample patches; it is not a fixed physiological duration.

| Recording | Encoder windows | Elapsed intervals |
| --- | --- | --- |
| 10 s at 125 Hz, 1,250 samples | One × 1,250 samples | `[0,10]` seconds |
| 6 s at 65 Hz, 390 samples | One × 390 samples | `[0,6]` seconds |
| 100 s at 200 Hz, 20,000 samples | Two × 12,800 samples | `[0,64]`, `[36,100]` seconds |

SPOTR z-scores each channel using its real samples, then pads internally with zeros to a multiple of 200. The numerical minimum is two samples because the upstream standard deviation uses sample correction. At most 128 channels enter a group. Defaults cap each channel at 1,000,000 samples and each request at 128 channels/128 global embeddings. Equal-shape batches contain at most `128*12800` waveform measurements. Over-budget requests fail explicitly.

## Physiological scope

The recording duration remains available through elapsed start/end encoding. One 512-dimensional global token represents each group/window. SPOTR's z-score removes absolute level and amplitude: preserving temperature or voltage during cleaning does not preserve it through this encoder. Questions requiring those quantities cannot rely on these normalized embeddings alone.

Acceptance does not establish suitability for every question. Pulse shape, pulse rate, respiration and HRV require different evidence and durations. NeuroKit recommends minutes of high-quality data for several frequency-domain HRV measures. SPOTR's published downstream pipeline uses 200 Hz and evaluates EEG, iEEG, ECG and PPG. Paya keeps the user's native rate and nine-type input contract; generalization and calibration must be evaluated by task, duration, rate and domain.

References: [SPOTR implementation](https://github.com/5GYYYYY/SPOTR), [SPOTR paper](https://arxiv.org/abs/2606.21973), [NeuroKit custom processing](https://neuropsychology.github.io/NeuroKit/examples/bio_custom/bio_custom.html), [HRV duration requirements](https://neuropsychology.github.io/NeuroKit/functions/hrv.html#hrv-frequency).
