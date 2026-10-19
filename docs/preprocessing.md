# Signal preparation

`signal + metadata → native-rate NeuroKit cleaning → router + question → selected encoder formatting → frozen encoder`

[physiology.py](../paya/physiology.py) returns one cleaned waveform per physical channel and an audit. [preprocessing.py](../paya/preprocessing.py) validates the streams, checks encoder eligibility, and resamples/frames only after selection. Cleaning and formatting receive no question or choices.

## Native cleaning

Units are converted first: ECG to mV; EEG/EMG to µV; acceleration to m/s²; angular velocity to rad/s; EDA to µS; temperature to °C. PPG and respiration retain their declared measurement conventions. Filters below use NeuroKit zero-phase Butterworth SOS unless stated otherwise; `fs` is native sampling rate in Hz.

| Signal | Cleaned waveform passed to the router | Native quality observations |
| --- | --- | --- |
| ECG | DC-preserving low-pass `min(150,0.4*fs)` | 0.5 Hz high-pass detection copy; NeuroKit peaks without artifact correction and template quality when enough beats exist |
| PPG | DC-preserving low-pass `min(40,0.4*fs)` | Elgendi 0.5–8 Hz detection copy when `fs>16`; peaks and template quality |
| EDA | NeuroKit 3 Hz cleaning if `fs>6`; otherwise no added low-pass | 0.05 Hz phasic decomposition where coverage permits; SCR onsets and peaks |
| Respiration | DC-preserving low-pass `min(3,0.4*fs)` | Khodadad2018 detection copy when `fs>6`; respiratory peaks and troughs |
| EMG | Bandpass 20–`min(450,0.4*fs)`; requires `fs>50` | Rectified 8 Hz envelope and activation at native rate |
| EEG | DC-preserving low-pass `min(100,0.4*fs)` | Bad-channel flags for ≥4 aligned channels with matching rate/reference |
| Accelerometer/gyroscope | DC-preserving low-pass `min(40,0.4*fs)` | Range and flatline flags; signed axes/gravity preserved |
| Temperature | Three-observation median with edge context | Range and flatline flags; absolute level/trend preserved |

Detection copies are temporary calculations for the audit. Only the single level-preserving cleaned waveform is retained and passed to the router. The same waveform is the source for selected-encoder formatting.

Optional electrical `metadata.powerline_hz` accepts numeric `0`, `50` or `60`; no frequency is assumed. A declared frequency applies a second-order ±1 Hz band-stop, rejecting bands outside Nyquist. Supplied leads, references and axis conventions are retained; channels are neither dropped nor synthesized.

## Formatting after selection

Before routing, type, duration and dimension checks produce a two-entry eligibility mask. These checks allocate no resampled arrays or encoder frames. If neither encoder is eligible, the request fails explicitly. The router processes the native cleaned channels and question, masks unavailable encoder logits, and chooses one encoder.

Only then does formatting run for that encoder. Downsampling adds a fourth-order anti-alias low-pass at `0.4*target_rate`. Polyphase conversion uses linear boundary extension and restores the mean after centering. No per-window z-score is applied. Upsampling changes the grid, not the measured bandwidth.

| Selected encoder | Rate | Framing |
| --- | --- | --- |
| NormWear | 65 Hz | Complete 390-sample frames; stride 390; per-frame detrending and Gaussian smoothing, sigma 1.3 |
| Mantis ECG/EEG | 500 Hz | Complete 512-sample frames; stride 256 |
| Mantis accelerometer/gyroscope | 100 Hz | Same |
| Mantis PPG | 125 Hz | Same |
| Mantis EDA/respiration/temperature | 32 Hz | Same |
| Mantis EMG | 2000 Hz | Same |

A final overlapping complete frame covers any trailing interval. No sample padding enters either encoder. NormWear formatting is unavailable for temperature/raw EMG and requires at least six seconds. The 32 Hz Mantis inputs require 16 seconds. Eligibility applies to every channel in the request.

Defaults allow 64 physical channels, 1,000,000 samples per channel, four simultaneous NormWear channels, 640 Mantis channel/frame pairs and 1,024 encoded signal tokens. Five minutes of single-channel 500 Hz ECG uses 585 Mantis frames.

## Quality boundary

Audits retain native rates/units, raw range, flatline flags, processing methods, detector warnings and event times in seconds. ECG/PPG detections within 0.25 seconds of either boundary are recorded separately and excluded from template measurements. Detector failure preserves the waveform, since abnormal physiology may be relevant to the question.

Peaks, quality scores, EMG envelopes and audit metadata are not CNN/MLP or decision-head input features. Template quality is not decision confidence. Full HRV tables, ECG delineation and ICA are not computed per prediction. Support for a signal type alone does not establish accuracy on every possible question.

References: [NeuroKit custom processing](https://neuropsychology.github.io/NeuroKit/examples/bio_custom/bio_custom.html), [ECG](https://neuropsychology.github.io/NeuroKit/functions/ecg.html), [PPG](https://neuropsychology.github.io/NeuroKit/functions/ppg.html), [EDA](https://neuropsychology.github.io/NeuroKit/functions/eda.html), [respiration](https://neuropsychology.github.io/NeuroKit/functions/rsp.html), [EMG](https://neuropsychology.github.io/NeuroKit/functions/emg.html).
