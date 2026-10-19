# Paya

The first non-autoregressive decision engine for body signals.** Typed decisions over PPG, ECG, EEG, accelerometer and gyroscope signals in a single forward pass ~20–70 ms.

## Use

```python
from paya.inference import load_bundle

model = load_bundle("artifacts/paya-v1")
state = {"signals": [{
    "samples": ppg_samples,
    "metadata": {
        "sensor_type": "ppg",
        "sampling_rate_hz": 125,
        "duration_s": 10,
        "signal_condition": "processed",
    },
}]}
result = model.predict(
    state=state,
    question="Which pulse-rate category describes this recording?",
    criteria=["Slow", "Normal", "Fast"],
)
```

Illustrative output:

```json
{"choice": "Normal", "probabilities": [0.1, 0.8, 0.1], "confidence": 0.8}
```

Each channel requires `sensor_type`, `sampling_rate_hz` and `duration_s`. Rates and durations must be finite positive numbers, with `duration_s = sample_count / sampling_rate_hz`. Supported types are ECG, EEG, accelerometer, gyroscope, PPG, EDA, respiration, temperature and EMG. Unit and lead fields are not used; samples retain their supplied amplitude scale. Channels must cover the same measured interval; different rates require explicit matching `start_time_s` values.

`signal_condition` accepts `raw` (default) or `processed`. Raw channels receive native-rate NeuroKit cleaning. Processed channels bypass NeuroKit. Neither path resamples the signal or applies an additional amplitude normalization.

## Architecture

`signal + metadata → validate → clean raw channels only → native variable-length formatting → frozen MantisV2 → shared decision head`

Frozen BioLORD-2023 encodes the question and each choice separately for the head. Only the decision head is trainable. It uses text self-attention, signal cross-attention, feed-forward layers and a shared candidate scorer.

Formatting retains the supplied recording context. A length divisible by 32 is encoded once; otherwise, an overlapping prefix/suffix pair covers every sample. For 10 seconds at 125 Hz, Mantis receives `[2,1,1248]` and returns `[2,512]`. Six seconds at 65 Hz becomes `[2,1,384] → [2,512]`. Elapsed interval endpoints accompany the signal embeddings. No router or output-width adapter is present.

The unmodified Mantis statistics require at least 64 native samples. This numerical minimum does not establish physiological sufficiency for a question. Resource budgets and raw-filter requirements also apply. See [architecture and dimensions](docs/architecture-fix.md), [signal preparation](docs/preprocessing.md) and [training and W&B](docs/training.md).

## Install and verify

Use Python 3.11 or newer:

```powershell
python -m pip install -e ".[test]"
python -m paya.assets --mantis third-party/mantis
python -m pytest -q -m "not integration" --basetemp cache/pytest-unit
python -m pytest -q -m integration --basetemp cache/pytest-integration
```

Mantis and BioLORD need local assets and `source.json` checksum records at the paths in [config/paya.yaml](config/paya.yaml). The integration test exercises real frozen weights and writes `artifacts/architecture-shapes.json`.

Paya versions are defined in [paya/provenance.py](paya/provenance.py): `v1`, Python package `1.0.0`. Accuracy, calibration, generalization and latency require trained weights and independent evaluation.
