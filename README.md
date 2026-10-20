# Paya

**The first non-autoregressive decision engine for body signals.** Typed decisions from physiological signals, a question and candidate answers.

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

Each channel requires `sensor_type`, `sampling_rate_hz` and `duration_s`. Rates and durations must be finite positive numbers, with `duration_s = sample_count / sampling_rate_hz`. Supported types are **EEG, iEEG, ECG and PPG**, case-insensitive. Unit and lead fields are not used. All channels in a recording must have the same sampling rate, sample count and start time.

`signal_condition` accepts `raw` (default) or `processed`. Raw channels receive native-rate NeuroKit cleaning. Processed channels bypass NeuroKit. Neither path resamples the signal. SPOTR then applies its pretrained per-channel z-score normalization internally.

## Architecture

`signal + metadata → validate → clean raw channels only → stack channels → frozen SPOTR once → shared decision head`

Frozen BioLORD-2023 encodes the question and each choice separately for the head. Only the decision head is trainable. It uses text self-attention, signal cross-attention, feed-forward layers and a shared candidate scorer.

The entire recording enters SPOTR once as `[1,C,L]`, in supplied channel order, and returns one global `[1,512]` embedding. Ten seconds of PPG at 125 Hz enters as `[1,1,1250]`; six seconds at 65 Hz enters as `[1,1,390]`. There is no grouping, splitting or overlap. Internal zero padding follows the pretrained implementation; the head receives the real duration through elapsed `[0,duration_s]` encoding.

Inputs must contain **1–128 channels and 2–12,800 samples per channel**. Longer recordings and mismatched channel rates are rejected. Raw filters can require more samples. Acceptance does not establish physiological sufficiency. SPOTR removes absolute amplitude and level, and its published downstream preparation uses 200 Hz; native-rate accuracy requires task-specific evaluation. See [architecture and dimensions](docs/architecture-fix.md), [signal preparation](docs/preprocessing.md) and [training and W&B](docs/training.md).

## Install and verify

Use Python 3.11 or newer:

```powershell
python -m pip install -e ".[test]"
python -m paya.assets --spotr third-party/spotr
python -m pytest -q -m "not integration" --basetemp cache/pytest-unit
python -m pytest -q -m integration --basetemp cache/pytest-integration
```

SPOTR and BioLORD need local assets and `source.json` checksum records at the paths in [config/paya.yaml](config/paya.yaml). The SPOTR asset command verifies the pinned upstream checkpoint before converting it to safetensors. The integration test exercises real frozen weights and writes `artifacts/architecture-shapes.json`.

Paya versions are defined in [paya/provenance.py](paya/provenance.py): `v1`, Python package `1.0.0`. Accuracy, calibration, generalization and latency require trained weights and independent evaluation.
