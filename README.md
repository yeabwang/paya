# Paya

**A non-autoregressive System 1 decision model for physiological signals.**

Paya takes a physiological recording, a question and candidate answers. Frozen signal and text encoders feed a trained MLP fusion head with Hadamard product features. A single forward pass returns the selected answer, its confidence and a probability for each candidate.

![Headline results for the selected Paya checkpoint](doc/figures/system.png)

These results describe **Paya full-003, update 4,797, temperature 1**, evaluated on 323,944 PPG QA rows. AUROC is the mean of task macro AUROCs; ECE uses 15 confidence bins. Latency is the median batch-one public-API time on an RTX 2080 Ti with warm text embeddings and uncached signal encoding. See the [evaluation protocol and limitations](doc/results.md).

## Results by task

![Per-task accuracy, AUROC and ECE](doc/figures/per-task.png)

All three panels use the same task ordering and the Okabe–Ito palette. Values are displayed as percentages; lower ECE is better. The high aggregate accuracy coexists with poor recall for several minority classes. The [results report](doc/results.md) includes task metrics, class coverage and latency conditions.

## Use

The verified model runtime uses **Python 3.12**. Install the recorded dependencies and the local package:

```bash
python -m pip install -r requirements-model.txt
python -m pip install -e .
```

Download the complete private [Paya inference bundle](https://huggingface.co/yeabwang/Paya) into `models/export/`. Access requires an authorized Hugging Face account. The bundle contains the decision checkpoint, both frozen encoders and the manifest needed by the loader.

```python
from paya.inference import load_model

model = load_model("models/export", device="cpu")

# ppg_samples: the 1,250 measurements from a processed 10-second PPG segment.
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
    question="What is the heart rate category?",
    criteria=["bradycardia", "normal", "tachycardia"],
)
# result: {"choice": ..., "confidence": ..., "probabilities": {...}}
```

Supply the actual waveform in `ppg_samples`. Use `device="cuda"` for a compatible GPU installation. CPU inference is supported; the published latency measures GPU inference. `signal_condition="processed"` bypasses physiological cleaning; raw inputs use the native-rate cleaning path. Neither path resamples the recording. The API accepts PPG, ECG, EEG and iEEG; the reported model evaluation covers **PPG only**.

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](doc/architecture.md) | Input contract, processing flow, encoder internals, tensor dimensions and complete decision-head layers |
| [Experiments](doc/experiments.md) | Development protocol, controlled comparisons, full-run recipe and checkpoint selection |
| [Fine-tuning](doc/finetune.md) | Eight continuation experiments, objectives, retention criteria and outcomes |
| [Results](doc/results.md) | System and task metrics, definitions, calibration comparison, latency and coverage |
| [Model card](models/model.card.md) | Model description, intended scope, training summary and checkpoint identity |

Exact evaluation values are in [doc/metrics.json](doc/metrics.json). The [checkpoint record](models/checkpoint.json), [runtime versions](models/runtime.json) and [verification record](models/verification.json) identify the released artifact.

## Repository layout

```text
paya/                 Supported model, input, training and inference code
tests/                Unit tests and pinned-encoder integration test
config/paya.yaml      Selected model and training settings
doc/                  Technical documentation, figures and evaluation metrics
models/               Model card and artifact identity records
```

Historical implementations and experiment tools are kept in ignored `retired/`; local shell helpers stay in ignored `.temp/`. Weights, datasets, feature caches, credentials and `models/export/` are excluded from Git. The standalone head checkpoint requires its matching frozen encoder assets and preparation identity.

For development, install the test extra and run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

The integration test requires the pinned pretrained assets. Previously recorded release verification passed 155 local unit tests and the 156-test GPU suite, including that integration test.
