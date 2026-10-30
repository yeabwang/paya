# Paya

**A non-autoregressive System 1 decision model for physiological signals.** Frozen SPOTR provides global, channel and temporal signal tokens. Frozen BioLORD encodes the question and candidates separately. A **480,257-parameter attention head** applies question/candidate self-attention, then cross-attention to the signal tokens, followed by a shared candidate scorer.

![1792750160494](image/README/1792750160494.webp)

## Setup and training

Use Python 3.12 and the recorded model dependencies:

```bash
python -m pip install -r requirements-model.txt
python -m pip install -e ".[test]"
```

Place the pinned SPOTR and BioLORD assets in `third-party/spotr` and `third-party/biolord`. Supply independent train, selection and calibration manifests at the paths in `config/paya.yaml`.

```bash
python -m paya.train --config config/paya.yaml --run-dir runs/attention --device cuda
python -m paya.calibrate --config config/paya.yaml --checkpoint runs/attention/best.pt --output runs/attention/calibrated.pt --device cuda
```

## Inference after training

Export a selected raw or independently calibrated checkpoint:

```bash
python -m paya.export --config config/paya.yaml --checkpoint runs/attention/calibrated.pt --output models/export
```

```python
from paya.inference import load_model

model = load_model("models/export", device="cpu")
state = {"signals": [{
    "samples": ppg_samples,
    "metadata": {
        "sensor_type": "ppg",
        "sampling_rate_hz": 125,
        "duration_s": len(ppg_samples) / 125,
        "signal_condition": "processed",
    },
}]}
result = model.predict(
    state, "What is the heart rate category?",
    ["bradycardia", "normal", "tachycardia"],
)
# {"choice": ..., "confidence": ..., "probabilities": {candidate: probability, ...}}
```

The example sampling rate describes this recording; it is not a fixed model rate. Confidence is the chosen answer's softmax probability. Calibration and accuracy must be measured on the intended dataset.

## Documentation


| Document                                    | Contents                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [Architecture](doc/architecture.md)         | Token layout, masking, attention blocks, input boundaries and scorer                    |
| [Training and fine-tuning](doc/finetune.md) | Dataset schema, fresh training, weight initialization, recovery, calibration and export |
| [Architecture decision](doc/experiments.md) | Promotion scope and retained evidence                                                   |

```bash
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

Integration tests require the pinned encoder assets.
