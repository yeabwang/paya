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

![System results](../docs/figures/system.png)

## Model

Frozen SPOTR provides a 512-dimensional signal representation. Frozen BioLORD-2023 encodes the question and candidates separately into 768-dimensional vectors. A 264,065-parameter MLP fusion head with Hadamard product features performs candidate scoring. Only this head was trained. Raw inputs receive native-rate physiological cleaning; processed inputs bypass it.

## Training

The full run used 1,637,289 processed PPG QA rows, three epochs, batch 512 and AdamW with learning rate 0.0003, weight decay 0.01 and label smoothing 0.05. The selected checkpoint is update 4,797, chosen by NLL on 196,707 selection rows. Separate calibration and test roles contain 21,342 and 323,944 rows. Known connected subject, recording and waveform identities remain disjoint. The release uses the original raw checkpoint at temperature 1; later continuation pilots did not qualify for replacement.

## Results


| Metric                |             Value |
| ----------------------- | ------------------: |
| Accuracy              |            76.86% |
| Mean task macro AUROC |           0.67465 |
| ECE, 15 bins          |           0.02115 |
| Mean task macro F1    |           0.37885 |
| NLL / Brier           | 0.65148 / 0.34587 |
| Median / p95 latency  |  12.30 / 15.81 ms |

![Per-task results](../docs/figures/per-task.png)

Scores use the same 323,944-row retrospective historical test. AUROC averages supported one-versus-rest classes within tasks, then tasks equally. Latency uses this exact checkpoint: 1,300 synchronized batch-one public-API requests on an RTX 2080 Ti, warm text and uncached signal encoding. Process/network overhead and concurrency are excluded.

## Scope

Reported performance covers processed, normalized 10-second PPG at 125 Hz. The API also accepts ECG, EEG and iEEG, but these scores do not validate those modalities. Rare-class coverage and recall remain limited, including AF, hypoxemia and severe sleep breathing. Subject-correlated QA rows are not independent observations; no significance or clinical deployment claim is made.

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
