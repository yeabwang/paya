# Paya

Paya makes structured decisions from body signals. Given a signal window, its metadata, a separate question and ordered candidate answers, it returns `choice`, `probabilities` and `confidence` without generating text.

## Use

```python
from paya.inference import load_bundle

model = load_bundle("artifacts/paya-v1")
state = {"signals": [{
    "samples": ecg_samples,
    "metadata": {"sensor_type": "ecg", "sampling_rate_hz": 500,
                 "unit": "mV", "lead": "II"},
}]}
result = model.predict(
    state=state,
    question="Which heart-rate category describes this signal?",
    criteria=["Slow heart rate", "Normal heart rate", "Fast heart rate"],
)
```

Example response:

```json
{"choice": "Normal heart rate", "probabilities": [0.1, 0.8, 0.1], "confidence": 0.8}
```

Probabilities follow the supplied criteria order. Confidence is the chosen answer's probability. Each channel requires finite samples, a unit, a supported type and a finite positive numeric sampling rate. Supported types are ECG, EEG, accelerometer, gyroscope, PPG, EDA, respiration, temperature and EMG; unknown or empty types and rates are rejected. Different native rates require an explicit shared `start_time_s` and matching measured intervals.

## Architecture

`signal + metadata → native-rate NeuroKit cleaning → question-conditioned router → selected encoder formatting → one frozen signal encoder → fixed width adapter → shared decision head`

- **Frozen:** NormWear, MantisV2 and BioLORD-2023. BioLORD encodes the question and each choice separately.
- **Trainable:** the waveform/question router and the shared head's self-attention, signal cross-attention, feed-forward layers and candidate scorer.
- One cleaned waveform per physical channel enters the router. Resampling and framing run only for the selected encoder. Metadata controls cleaning, compatibility checks and that formatting. The head receives waveform vectors, text vectors, relative times and padding masks. Output adapters preserve NormWear's 768 dimensions and append 256 zeros to Mantis's 512 dimensions.

See [architecture and layer dimensions](docs/architecture-fix.md), [signal preparation](docs/preprocessing.md) and [training and W&B](docs/training.md).

## Install and verify

Use Python 3.11 or newer:

```powershell
python -m pip install -e ".[test]"
python -m paya.assets --mantis third-party/mantis
python -m pytest -q -m "not integration" --basetemp cache/pytest-unit
python -m pytest -q -m integration --basetemp cache/pytest-integration
```

All three encoders need local assets and `source.json` checksum records under the paths in [config/paya.yaml](config/paya.yaml). The integration test exercises real frozen weights and records shapes in `artifacts/architecture-shapes.json`.

Paya's package and artifact versions are defined in [paya/provenance.py](paya/provenance.py): first release `v1` (Python package `1.0.0`). Upstream model names and pinned revisions retain their published identifiers. Decision accuracy, calibration and latency require trained weights and evaluation on independent recordings.
