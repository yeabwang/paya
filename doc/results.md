# Results

All release figures and values below refer to **Paya full-003, update 4,797, temperature 1**, with no post-hoc calibration. The historical test contains **323,944 QA rows across 13 tasks**. The matching checkpoint is identified in [models/checkpoint.json](../models/checkpoint.json); exact metrics and confusion matrices are preserved in [metrics.json](metrics.json).

![Headline results for the selected Paya checkpoint](figures/system.png)

## System results

| Metric | Value | Aggregation |
| --- | ---: | --- |
| Accuracy | 76.8599% | QA-row weighted |
| Macro AUROC | 0.674655 | Supported-class macro within each task, then equal-weight mean of tasks |
| ECE, 15 bins | 0.021148 | Pooled top-label confidence |
| Macro F1 | 0.378854 | Candidate-class macro within each task, then equal-weight mean of tasks |
| NLL | 0.651483 | QA-row mean |
| Brier | 0.345870 | Sum over candidate classes, then QA-row mean |
| Median latency | 12.302 ms | Batch-one public API |
| p95 / p99 latency | 15.809 / 17.577 ms | Same measurement |
| Mean latency | 12.650 ms | Same measurement |

The headline graphic rounds these values to 76.86%, 0.6747, 0.0211 and 12.30 ms. Those four values describe the same checkpoint variant. They do not use the earlier calibrated-model latency measurement.

## Results by task

![Per-task accuracy, AUROC and ECE](figures/per-task.png)

The panels share descending accuracy order. Accuracy is blue (`#0072B2`), AUROC orange (`#E69F00`) and ECE green (`#009E73`) from the Okabe–Ito palette. Figure labels express all three metrics as percentages; tables retain AUROC and ECE on the 0–1 scale. A vector version of the task figure is available as [PDF](figures/per-task.pdf).

| Task | QA rows | Accuracy | Macro AUROC | ECE |
| --- | ---: | ---: | ---: | ---: |
| SpO₂ | 4,492 | 97.75% | 0.4548 | 0.1134 |
| Heart rate | 78,730 | 89.97% | 0.9211 | 0.0619 |
| AF detection | 32,560 | 85.56% | 0.5788 | 0.0851 |
| HRV SDNN | 41,709 | 84.65% | 0.7186 | 0.0578 |
| Signal quality | 9,427 | 83.19% | 0.8972 | 0.0204 |
| HRV pNN50 | 41,709 | 77.88% | 0.6760 | 0.0533 |
| HRV RMSSD | 41,709 | 71.96% | 0.6717 | 0.0690 |
| Arrhythmia | 5,243 | 66.76% | 0.8812 | 0.0255 |
| Blood pressure | 23,138 | 62.45% | 0.5899 | 0.1205 |
| Sleep breathing | 29,754 | 54.45% | 0.4359 | 0.1125 |
| Stress | 597 | 46.40% | 0.6572 | 0.0525 |
| Respiratory rate | 9,556 | 40.18% | 0.5458 | 0.2138 |
| Activity | 5,320 | 37.37% | 0.7423 | 0.0934 |

High accuracy on an imbalanced task can coexist with poor minority recall and low AUROC. Conversely, activity and arrhythmia show useful ranking relative to their top-choice accuracy. The decision rule remains the highest-probability candidate; no test-derived class thresholds were fitted. Task ECE can exceed pooled system ECE because calibration errors can offset when tasks are combined. The pooled value is not the mean of the task values.

## Metric definitions

For row `i`, let `p_i,k` be the probability of real candidate `k`, `y_i` the correct index and `ŷ_i` the selected index. Padding is excluded. Candidate order is aligned by answer string when aggregating classes.

| Metric | Definition and conventions |
| --- | --- |
| Accuracy | Mean of `1[ŷ_i = y_i]` over QA rows |
| NLL | Mean of `−log(p_i,y_i)`; natural logarithm |
| Brier | Mean of `Σ_k (p_i,k − 1[k=y_i])²`; not divided by candidate count |
| ECE | Split top-choice confidence into 15 equal-width bins; sum `bin_fraction × abs(bin_accuracy − bin_mean_confidence)` over nonempty bins |
| Class AUROC | One-versus-rest ranking using the corresponding candidate probability; ties receive half credit |
| Task macro AUROC | Mean class AUROC over classes with both positive and negative examples; undefined classes are omitted, not assigned zero |
| Task macro F1 | Mean class F1 over the task's candidate vocabulary, including zero-F1 classes with no correct predictions |
| System macro AUROC / F1 | Equal-weight mean of the corresponding task macro values |

Accuracy, NLL, Brier and ECE pool QA rows. There is no single pooled AUROC across unrelated task vocabularies. Moderate sleep breathing has zero test positives, so its class AUROC is undefined; sleep-breathing macro AUROC averages the other three supported classes. Macro F1 still includes all four candidates. The implementation is in [evaluate.py](../paya/evaluate.py) and [classification.py](../paya/classification.py).

## Selection and calibration

The checkpoint was selected by minimum NLL on **196,707 selection rows** at update 4,797. A native-input reevaluation of the release produced:

| Selection metric | Value |
| --- | ---: |
| Accuracy | 69.5735% |
| NLL | 0.761795 |
| Brier | 0.429431 |
| ECE | 0.079107 |

These are selection results, not test results. Population, task and class prevalence differ by role; the difference from test accuracy is not a training-accuracy measurement.

### Calibration

A single positive temperature was fitted on the independent **21,342-row calibration role** after checkpoint selection. It rescales logits before softmax without changing the highest-scoring answer. The fitted temperature was **1.048308**; calibration-role NLL decreased from 0.720900 to 0.720216.

| Historical test metric | Raw, T = 1 | Calibrated, T = 1.048308 |
| --- | ---: | ---: |
| NLL | **0.651483** | 0.651503 |
| Brier | **0.345870** | 0.346460 |
| ECE | **0.021148** | 0.025102 |

The small fitting-set improvement did not transfer to these historical test probability scores. Accuracy and F1 were unchanged. The released artifact and figures retain **T = 1**. Calibration covered only 10 of 13 tasks: activity, AF detection and stress had no calibration examples. A global fit is therefore not evidence of task-specific calibration on those tasks.

## Latency protocol

The release checkpoint was benchmarked on an otherwise idle **single NVIDIA RTX 2080 Ti**, with PyTorch **2.8.0+cu128** and CUDA **12.8**. The measurement used **1,300 batch-one public `predict` calls**, a seeded request order and 13 native processed PPG inputs covering the 13 task types. Each recording was 10 seconds at 125 Hz.

CUDA work was synchronized around measurement. The model and exact text embeddings were warm; **signal-feature caching was disabled**. Timing included input validation/preparation, signal encoding, head scoring and returned probabilities. Process startup, initial model loading, first-use text encoding, network/service overhead and concurrent serving were excluded. These numbers describe warm local inference under those conditions; they are not cold-start or loaded-service measurements.

The 12.302 ms median belongs to this exact raw checkpoint. The earlier 11.915 ms value came from a different calibrated-checkpoint benchmark and is not substituted for the release result. Historical training throughput and cache benchmarks are also distinct from batch-one inference latency.

## Class coverage and failure analysis

| Finding | Matching test evidence | Implication |
| --- | --- | --- |
| Low AF sensitivity | 241 / 4,789 AF examples correct; recall 5.03% | 85.56% task accuracy is dominated by non-AF performance |
| Hypoxemia not recovered | 0 / 89 correct: 84 mild, 1 moderate, 4 severe | 97.75% SpO₂ accuracy does not demonstrate hypoxemia detection |
| Severe sleep breathing not recovered | 0 / 2,219 correct | Poor sensitivity persists despite nonzero support |
| Moderate sleep breathing unmeasurable | No test positives | Recall and class AUROC cannot be estimated |
| Additional zero-recall labels | Signal-quality `symmetric_unusual`, respiratory `tachypnea`, and HRV pNN50 `normal`, among others | Aggregate averages conceal class-specific failures |

The inherited validation corpus has no severe sleep-breathing examples and only one moderate-hypoxemia example. Selection and calibration therefore cannot support precise conclusions for all test classes. Stress has just 597 test rows, and several rare labels have much smaller support. [metrics.json](metrics.json) retains every class support, confusion matrix and supported-AUROC count.

QA rows can share subjects, recordings and waveforms. Their count is not an independent-patient sample size; no subject-bootstrap confidence interval or significance claim is supplied. This is a retrospective historical benchmark, and broader development was informed by earlier test inspection. QA mapping checks establish fidelity to supplied labels, not independent clinical verification of those labels. Reported performance covers processed PPG, not ECG, EEG or iEEG.

## Artifact identity and verification

| Record | Value |
| --- | --- |
| Run / selected update | `paya-full-003` / `4797` |
| Temperature / calibrated | `1.0` / `false` |
| Checkpoint SHA-256 | `461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba` |
| Learned-head SHA-256 | `e869cfe0db5df2e14d7ec815d8dc504bd3bfa1680500432820650b754c5dd746` |
| Training/evaluation revision | `48754f3b86cb0c29e04673acfefd6f2f6418f55e` |

Previously recorded verification passed 155 local unit tests and all 156 GPU tests, including pinned pretrained integration. Repository cleanup preserved probabilities exactly on 13 real public inputs: maximum difference **0.0**. The complete inference export reloaded with the same learned-weight identity. These are release verification results, not newly rerun GPU tests for the documentation update. See the [verification record](../models/verification.json) and [model card](../models/model.card.md).
