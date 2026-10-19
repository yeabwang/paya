# Paya

**A non-autoregressive System 1 decision model for body signals.** Given a signal window, its metadata, a separate question, and candidate answers, Paya returns a choice, ordered probabilities, and confidence through one shared decision model.

The replacement architecture is implemented and ready for review. The router and decision head need fresh training; no earlier accuracy, calibration, or latency results apply to this model.

## Architecture

`signal + metadata → NeuroKit preparation → question-conditioned router → selected frozen encoder → fixed width adapter → shared decision head`

- Frozen signal encoders: **NormWear** and **MantisV2**. Frozen text encoder: **BioLORD-2023**.
- Preprocessing prepares compatible waveform views before routing. Inference executes one signal encoder per request.
- The trainable router uses small shared waveform convolutions and the question embedding. Metadata configures preprocessing and never becomes a learned input vector.
- NormWear vectors retain width 768. Mantis vectors retain their original 512 coordinates and append 256 zeros. These adapters have no parameters.
- The trainable head uses two blocks of bidirectional self-attention, signal cross-attention, and feed-forward layers, followed by one shared candidate scorer.

See [the full specification](docs/architecture-fix.md) for dimensions and [implementation review notes](docs/implementation-review.md) for the checks performed and current limits.

## Public input and output

```python
from paya import PayaModel
from paya.checkpoint import load_checkpoint

model = PayaModel.from_assets({
    "normwear": "third-party/normwear",
    "mantis": "third-party/mantis",
    "text": "third-party/biolord",
})
# After retraining and calibration, explicitly load the new checkpoint:
load_checkpoint("runs/body-v2/calibrated.pt", model)

state = {"signals": [{
    "samples": ecg_samples,  # one-dimensional measurements
    "metadata": {
        "sensor_type": "ecg", "sampling_rate_hz": 500,
        "unit": "mV", "lead": "II",
    },
}]}
question = "Which heart-rate category describes this signal?"
criteria = ["Slow heart rate", "Normal heart rate", "Fast heart rate"]
result = model.predict(state=state, question=question, criteria=criteria)
```

Illustrative output, not a measured prediction:

```json
{"choice": "Normal heart rate", "probabilities": [0.1, 0.8, 0.1], "confidence": 0.8}
```

`probabilities[i]` belongs to `criteria[i]`; `confidence` equals the chosen candidate's probability. Question, each criterion, and the internal fixed ending are independently encoded. There is no generated explanation or separate confidence head.

Every channel requires a type, a finite positive numeric rate, a unit, and finite samples. Supported types are `ecg`, `eeg`, `accelerometer`, `gyroscope`, `ppg`, `eda`, `respiration`, `temperature`, and `emg`. Unknown/empty types and rates are rejected. Boolean and string rates are rejected. Heterogeneous rates require a shared explicit `start_time_s`. Signals must cover the same physical interval; short or oversized requests without a compatible view are rejected.

The default budget is 64 physical channels, 1,000,000 samples per channel, 1,024 encoded signal tokens, four simultaneous NormWear channels, and 128 Mantis channel/frame pairs. NormWear needs six seconds of coverage. Mantis uses complete 512-sample frames; at 32 Hz, temperature/EDA/respiration require at least 16 seconds. Upsampling adds no new measurement evidence. Nine input types do not establish competence on every possible question.

## Install and check shapes

Use Python 3.11 or newer and install the project with its test dependencies:

```powershell
python -m pip install -e ".[test]"
python -m paya.assets --mantis third-party/mantis
python -m pytest -q -m "not integration" --basetemp cache/pytest-unit
python -m pytest -q -m integration
```

The integration check requires all three local encoder asset directories, including their `source.json` checksum records. NormWear and BioLORD assets already exist in this workspace. Mantis is downloaded at the fixed revision recorded in the specification. There is no automatic loading of old trained Paya checkpoints or feature caches.

The integration check runs synthetic waveforms through the actual frozen weights and records tensor shapes in `artifacts/architecture-shapes.json`. It also checks gradient boundaries without updating those weights. This is an implementation check, not model-performance evaluation.

## Retraining workflow after review

Each fresh JSONL dataset row has separate `state`, `question`, ordered `criteria`, integer zero-based `target`, and audit-only `source_id`, `group_id`, and `recording_id`. All questions/windows from a recording stay in the same split. The loader checks separation of these identifiers across the configured splits.

Set the new dataset paths and budgets in [config/paya.yaml](config/paya.yaml). The following commands are available for the next stage; they have not been run on a real dataset:

```powershell
python -m paya.train --config config/paya.yaml --run-dir runs/body-v2 --device cpu
python -m paya.calibrate --config config/paya.yaml --checkpoint runs/body-v2/best.pt --output runs/body-v2/calibrated.pt
python -m paya.evaluate --config config/paya.yaml --checkpoint runs/body-v2/calibrated.pt --split test --output runs/body-v2/test.json
python -m paya.export --config config/paya.yaml --checkpoint runs/body-v2/calibrated.pt --output artifacts/body-v2
```

Training first balances compatible experts while warming up the shared head, then trains the gate from training-only expert NLL targets and trains the head under actual top-1 dispatch. Only routed epochs can produce `best.pt`. Calibration fits one global temperature on a separate dataset after freezing the learned policy. Checkpoints bind the weights, preparation, frozen assets, and calibration; incompatible old artifacts are rejected.

Load an exported bundle with `from paya.inference import load_bundle`, then use the same `predict` interface. The retired implementation is preserved outside the project at `C:\jev\retired\paya-20261019-architecture-v1`.
