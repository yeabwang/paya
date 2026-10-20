# Signal preparation

`validate → clean raw channels only → stack complete recording → frozen SPOTR once`

## Input contract

Each physical channel contains a one-dimensional `samples` array and `metadata`:

| Metadata | Requirement |
| --- | --- |
| `sensor_type` | EEG, iEEG, ECG or PPG; case-insensitive |
| `sampling_rate_hz` | Required finite positive number, never inferred or replaced |
| `duration_s` | Required finite positive number matching `sample_count / sampling_rate_hz` |
| `signal_condition` | `raw` by default, or `processed` |

Samples must be finite and representable in float32. Preprocessing retains their supplied amplitude scale; unit conversion and lead handling are absent. All channels must have the same sampling rate, sample count and start time. Optional timestamps must match the declared uniform rate. Supplied channel order is preserved for SPOTR's learned channel positions.

## Raw NeuroKit processing

[physiology.py](../paya/physiology.py) returns one native-rate cleaned waveform plus audit observations. These operations do not change sample count or sampling rate. `fs` below is the supplied rate.

| Signal | Retained waveform | Audit calculations |
| --- | --- | --- |
| ECG | DC-preserving low-pass `min(150,0.4*fs)` | High-pass detection copy; peaks and template quality |
| PPG | DC-preserving low-pass `min(40,0.4*fs)` | Elgendi detection copy where rate permits; peaks and template quality |
| EEG | DC-preserving low-pass `min(100,0.4*fs)` | Bad-channel flags when at least four raw scalp EEG channels share a reference |
| iEEG | DC-preserving low-pass `0.4*fs`, retaining the supplied intracranial reference | Range and flatline observations; scalp EEG outlier checks are excluded |

Optional electrical `powerline_hz` accepts numeric `0`, `50` or `60`; no frequency is assumed. A declared notch band must fit below native Nyquist. Detection copies and quality measurements are audit-only. Detector failure preserves the cleaned waveform and records unavailability rather than synthesizing normal physiology.

## Processed signals

`signal_condition: processed` skips all NeuroKit cleaning, peak detection and quality calculations. Validation and channel stacking still run. Provided samples enter SPOTR unchanged apart from float32 representation; its own pretrained normalization and internal padding apply to both raw and processed inputs.

## One native recording

Stack all channels as `[C,L]` and run SPOTR once as `[1,C,L]`. There is no grouping, splitting, overlap, resampling or external padding. Mismatched channel rates or lengths and recordings above 12,800 samples per channel are rejected before NeuroKit runs. The limit follows the checkpoint's 128 temporal positions and 100-sample patches; it is not a fixed physiological duration.

| Recording | SPOTR input for one channel | Elapsed interval |
| --- | --- | --- |
| 10 s at 125 Hz, 1,250 samples | `[1,1,1250]` | `[0,10]` seconds |
| 6 s at 65 Hz, 390 samples | `[1,1,390]` | `[0,6]` seconds |
| 64 s at 200 Hz, 12,800 samples | `[1,1,12800]` | `[0,64]` seconds |

SPOTR z-scores each channel using its real samples, then pads internally with zeros to a multiple of 200. The numerical minimum is two samples because the upstream standard deviation uses sample correction. Defaults cap each request at 128 channels and 12,800 samples per channel. Every accepted recording produces one 512-dimensional global embedding.

## Physiological scope

The recording duration remains available through elapsed `[0,duration_s]` encoding. These two numbers do not trigger another inference. SPOTR's z-score removes absolute level and amplitude: absolute voltage or optical intensity cannot be recovered from normalized embeddings alone.

Acceptance does not establish suitability for every question. Pulse shape, pulse rate and HRV require different evidence and durations. SPOTR's published downstream pipeline uses 200 Hz and evaluates EEG, iEEG, ECG and PPG. Paya supports those four signal types and keeps the user's native rate; generalization and calibration must be evaluated by task, duration, rate and domain.

References: [SPOTR implementation](https://github.com/5GYYYYY/SPOTR), [SPOTR paper](https://arxiv.org/abs/2606.21973), [NeuroKit custom processing](https://neuropsychology.github.io/NeuroKit/examples/bio_custom/bio_custom.html), [HRV duration requirements](https://neuropsychology.github.io/NeuroKit/functions/hrv.html#hrv-frequency).
