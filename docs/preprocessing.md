# Signal preparation

Preparation validates and converts physical streams, cleans them at their native rates, then constructs fixed waveform views for NormWear and MantisV2 before routing. It receives no question or candidate answers. [physiology.py](../paya/physiology.py) handles native cleaning and quality observations; [preprocessing.py](../paya/preprocessing.py) handles resampling, framing and budgets.

## Native processing

Units are converted first: ECG to mV; EEG/EMG to µV; acceleration to m/s²; angular velocity to rad/s; EDA to µS; temperature to °C. PPG and respiration retain their declared measurement conventions. Filters below use NeuroKit zero-phase Butterworth SOS unless stated otherwise; `fs` means native sampling rate in Hz.

| Signal | Broad waveform for Mantis | Dynamic waveform for NormWear | Native quality observations |
| --- | --- | --- | --- |
| ECG | DC-preserving low-pass `min(150,0.4*fs)` | Fifth-order 0.5 Hz high-pass | NeuroKit peaks without artifact correction; template quality when enough beats exist |
| PPG | DC-preserving low-pass `min(40,0.4*fs)` | Elgendi 0.5–8 Hz cleaning if `fs>16`; otherwise broad | Peaks and template quality where available |
| EDA | NeuroKit 3 Hz cleaning if `fs>6`; otherwise no added low-pass | 0.05 Hz phasic high-pass where coverage permits; otherwise broad | SCR onsets and peaks; tonic level retained in broad view |
| Respiration | DC-preserving low-pass `min(3,0.4*fs)` | Khodadad2018 cleaning if `fs>6`; otherwise broad | Respiratory peaks and troughs |
| EMG | Bandpass 20–`min(450,0.4*fs)`; requires `fs>50` | Unavailable | Rectified 8 Hz envelope and activation at native rate |
| EEG | DC-preserving low-pass `min(100,0.4*fs)` | Same cleaned waveform | Bad-channel flags for ≥4 aligned channels with matching rate/reference |
| Accelerometer/gyroscope | DC-preserving low-pass `min(40,0.4*fs)` | Same cleaned waveform | Range and flatline flags; signed axes/gravity preserved |
| Temperature | Three-observation median with edge context | Unavailable | Range and flatline flags; absolute level/trend preserved |

Optional electrical `metadata.powerline_hz` accepts numeric `0`, `50` or `60`; no frequency is assumed. A declared frequency applies a second-order ±1 Hz band-stop, rejecting bands outside Nyquist. Supplied leads, references and axis conventions are retained; channels are neither dropped nor synthesized.

## Encoder views

Downsampling adds a fourth-order anti-alias low-pass at `0.4*target_rate`. Polyphase conversion uses linear boundary extension and restores the mean after centering to preserve physical levels. No per-window z-score is applied. Upsampling changes the grid, not the measured bandwidth.

| View | Rate | Framing |
| --- | --- | --- |
| NormWear | 65 Hz | Complete 390-sample frames; stride 390; per-frame detrending and Gaussian smoothing, sigma 1.3 |
| Mantis ECG/EEG | 500 Hz | Complete 512-sample frames; stride 256 |
| Mantis accelerometer/gyroscope | 100 Hz | Same |
| Mantis PPG | 125 Hz | Same |
| Mantis EDA/respiration/temperature | 32 Hz | Same |
| Mantis EMG | 2000 Hz | Same |

A final overlapping complete frame covers any trailing interval. No sample padding enters either encoder. Encoder eligibility requires coverage for every channel in the request and compliance with budgets; if neither view is valid, preparation fails explicitly.

Defaults allow 64 physical channels, 1,000,000 samples per channel, four simultaneous NormWear channels, 640 Mantis channel/frame pairs and 1,024 encoded signal tokens. NormWear requires six seconds; the 32 Hz Mantis views require 16 seconds. Five minutes of single-channel 500 Hz ECG uses 585 Mantis frames. Frame counts are checked before allocating resampled arrays.

## Quality boundary

Audits retain native rates/units, raw range, flatline flags, processing methods, detector warnings and event times in seconds. ECG/PPG detections within 0.25 seconds of either boundary are recorded separately and excluded from template measurements. Detector failure preserves the waveform, since abnormal physiology may be relevant to the question.

Peaks, quality scores, EMG envelopes and audit metadata are not router/head input features. Template quality is not decision confidence. The pipeline does not compute full HRV tables, ECG delineation or ICA per prediction. The router learns which prepared view fits the question; support for a signal type alone does not establish accuracy on every possible question.

References: [NeuroKit custom processing](https://neuropsychology.github.io/NeuroKit/examples/bio_custom/bio_custom.html), [ECG](https://neuropsychology.github.io/NeuroKit/functions/ecg.html), [PPG](https://neuropsychology.github.io/NeuroKit/functions/ppg.html), [EDA](https://neuropsychology.github.io/NeuroKit/functions/eda.html), [respiration](https://neuropsychology.github.io/NeuroKit/functions/rsp.html), [EMG](https://neuropsychology.github.io/NeuroKit/functions/emg.html).
