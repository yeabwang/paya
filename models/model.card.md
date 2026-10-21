---
language: en
library_name: pytorch
tags:
  - physiological-signals
  - time-series
  - classification
  - non-autoregressive
---
# Paya

Paya is a **non-autoregressive System 1 decision model for physiological signals**. Given a signal recording, a question and candidate answers, it returns one choice and a probability for each candidate in a single forward pass.

![Headline results for the selected Paya checkpoint](../doc/figures/system.png)

## Model

Frozen SPOTR provides one 512-dimensional global signal vector. Frozen BioLORD-2023 encodes the question and each candidate independently into normalized 768-dimensional vectors. Only the **264,065-parameter MLP fusion head with Hadamard product features** was trained.

| Stage | Operation | Output width |
| --- | --- | ---: |
| Signal projection | LayerNorm(512), Linear(512,128), GELU | 128 |
| Shared question/candidate projection | Linear(768,128), LayerNorm(128) | 128 |
| Context MLP | Concatenate signal, question and their Hadamard product; Linear(384,128), GELU, Dropout(0.1) | 128 |
| Shared candidate scorer | Concatenate context, candidate and their Hadamard product; Linear(384,128), GELU, Dropout(0.1), Linear(128,1) | 1 per candidate |

Candidate logits are masked and converted to probabilities with softmax at temperature 1. The API returns the highest-probability choice, confidence and choice-keyed probabilities. This is one shared scorer across tasks, with no autoregressive decoder or task-specific output head. Raw inputs receive native-rate physiological cleaning; processed inputs bypass it. SPOTR normalizes channels and pads internally, without external resampling. See the [complete architecture](../doc/architecture.md) for encoder internals, tensor shapes and input limits.

## Training

The full run used 1,637,289 processed, normalized PPG QA rows, each containing a 10-second signal at 125 Hz. The three-epoch run used batch 512, AdamW with learning rate 0.0003, weight decay 0.01, label smoothing 0.05 and gradient clipping at norm 1. No class weighting was used. Both encoders remained frozen. Questions and targets preserve the supplied QA content.

The selected checkpoint is update 4,797 of 9,594, chosen by NLL on 196,707 selection rows at 1,599-update intervals. Separate calibration and test roles contain 21,342 and 323,944 rows. Native train/test membership was retained; the supplied validation corpus was divided into selection and calibration roles before training. Known connected subject, recording and waveform identities remain disjoint. The release uses the original raw checkpoint at temperature 1; later continuation pilots did not qualify for replacement. See the [experiment protocol](../doc/experiments.md) and [continuation experiment outcomes](../doc/experiments.md#checkpoint-continuation-experiments).

## Results


| Metric                |             Value |
| ----------------------- | ------------------: |
| Accuracy              |            76.86% |
| Mean task macro AUROC |           0.67465 |
| ECE, 15 bins          |           0.02115 |
| Mean task macro F1    |           0.37885 |
| NLL / Brier           | 0.65148 / 0.34587 |
| Median / p95 latency  |  12.30 / 15.81 ms |

![Per-task accuracy, AUROC and ECE](../doc/figures/per-task.png)

Scores use the same 323,944-row retrospective historical test. AUROC averages supported one-versus-rest classes within tasks, then tasks equally. ECE uses pooled top-label confidence in 15 equal-width bins; it is not mean task ECE. The task figure expresses all three metrics as percentages. Latency uses this exact checkpoint: 1,300 synchronized batch-one public-API requests on an RTX 2080 Ti, warm text and uncached signal encoding. Process/network overhead and concurrency are excluded. [Full results](../doc/results.md) include metric definitions, task scores, calibration comparison and class support; [metrics.json](../doc/metrics.json) retains exact values.

## Scope

The intended use is research on candidate-based physiological signal decisions. Reported performance covers processed, normalized 10-second PPG at 125 Hz. The API also accepts ECG, EEG and iEEG, but these scores do not validate those modalities or unseen task vocabularies.

AF recall is 5.03%; hypoxemia and severe sleep-breathing recall are 0/89 and 0/2,219. Moderate sleep breathing has no test positives. Calibration has no activity, AF or stress rows. Subject-correlated QA rows are not independent observations; no significance or clinical deployment claim is made. The prior historical test was already inspected during development, so this is not a fresh unseen evaluation.

## Files and use

`decision.pt` is the exact selected head checkpoint. The export also includes pinned signal/text encoder assets, source checksums and a matching runtime configuration. The verified runtime uses Python 3.12. Install `requirements-model.txt` and the included `code/` package; the loader verifies architecture, preparation and asset identities. The checkpoint alone requires these frozen assets.

```python
from paya.inference import load_model
model = load_model("models/export", device="cpu")
result = model.predict(state=state, question=question, criteria=criteria)
```

SPOTR and BioLORD-2023 retain their upstream attribution and licenses, included with their assets and vendor implementation. Paya's trained decision head and evaluation are identified separately.

Checkpoint SHA-256: `461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba`.
Training revision: `48754f3b86cb0c29e04673acfefd6f2f6418f55e`.
