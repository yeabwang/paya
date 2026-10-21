# Paya checkpoint-continuation experiment report

Eight bounded continuation pilots completed. None qualifies for expansion or
replacement. Every branch selected update 0: the original parent. All eight
eligible heads are tensor-identical to that parent, whose file SHA-256 is
`461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba`. No larger development run, new full run or NormWear
training had started at this stage. The monitoring timer remains paused. This
report completes the pilot stage before the separately authorized NormWear work;
that separate experiment must not alter or replace the current main model.

## Protocol and implementation

These are Protocol A comparisons on unchanged train/selection roles. Training
uses the same deterministic 64,000-row sample; selection monitoring uses the same
16,384 rows, with final comparisons over all 196,707 selection rows. Each branch
uses 400 updates, batch 512 and seed 42, giving 204,800 row draws. Replay is
task-balanced within the recovery/protected pools; it is not class balancing.
The recovery tasks are respiratory rate, sleep breathing, AF and blood pressure.
Rows repeat under sampling; draws are not independent examples or epochs.

The frozen signal/text encoders and the 264,065-parameter MLP fusion head with
Hadamard product features are unchanged. This cycle changes optimization and
sampling, not the inference architecture. Dropout is 0.1, label smoothing 0.05,
AdamW weight decay 0.01 and gradient clipping 1. Initial learning rate is 3e-5
except the declared 1e-5 branch. Replay assigns 75% of draws to recovery tasks
and 25% to other tasks. Weak-only assigns no replay.

L2-SP penalizes squared deviation from the parent, normalized by its squared
parameter norm. Distillation adds a temperature-2 KL penalty on protected-task
outputs alongside labeled replay. PCGrad projects opposing task gradients while
retaining their batch-frequency weighting. The mean pairwise gradient cosine was
negative for 62.82% of task pairs over three training-batch probes; that motivated
this diagnostic and does not prove causal harmful interference.

Checkpoints are evaluated every 100 updates. A plateau halves the learning rate
after two unsuccessful checks; stopping patience is five checks. The 400-update
cap allows four checks, so no pilot actually triggers patience-based stopping.
The controller's stopping/recovery behavior is covered by tests. Every branch
keeps both the best research checkpoint and a checkpoint satisfying retention.
The monitored objective is mean per-task NLL. Since no updated checkpoint beats
the initial objective, even the research selection remains the parent.

Implementation revisions: audit `c302d53`, first controls/pilots `f9b013e`,
preservation/new-role support `0a12b0f`, final two controls and reorganization
`3be95f2`. Per-file source/config/input bindings are preserved in the JSON receipt.
The final GPU suite passed **182 unit tests**, with one pretrained integration
test deselected. The focused local suite passed 25 tests. Forced recovery
reproduced dropout, optimizer, sampler and controller outcomes exactly, including
the PCGrad branch. Historical pretrained integration and this cycle's actual
native-feature/public-API checks are separate from that unit-suite result.

## Final weights versus parent

These rows evaluate each branch's actual update-400 weights, not its selected
update-0 copy. Accuracy/NLL/Brier/ECE pool QA rows. F1 averages all candidate
classes within each task, then tasks equally; an absent class has F1 zero.
AUROC averages supported one-versus-rest classes within each task, then tasks;
undefined classes remain null. ECE uses 15 equal-width confidence bins. These
are selection results, not new test results.

| Final trained weights | Accuracy | Macro F1 | AUROC | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged parent | 69.574% | 0.39028 | 0.72725 | 0.76180 | 0.42943 | 0.07911 |
| All-task continuation | 69.259% | 0.38589 | 0.72675 | 0.76874 | 0.43220 | 0.09048 |
| Weak-task-only continuation | 69.173% | 0.39035 | 0.72098 | 0.78603 | 0.44054 | 0.09226 |
| 25% replay | 69.310% | 0.38893 | 0.71679 | 0.77638 | 0.43559 | 0.09166 |
| Replay + L2-SP, coefficient 0.1 | 69.309% | 0.38893 | 0.71680 | 0.77637 | 0.43559 | 0.09165 |
| Replay + distillation, coefficient 0.1 | 69.389% | 0.38908 | 0.71875 | 0.77462 | 0.43471 | 0.08900 |
| Replay, learning rate 1e-5 | 69.386% | 0.38829 | 0.72292 | 0.77168 | 0.43426 | 0.09135 |
| Replay + PCGrad | 69.362% | 0.38936 | 0.71648 | 0.77709 | 0.43545 | 0.09453 |
| Replay + L2-SP, coefficient 10 | 69.318% | 0.38924 | 0.71777 | 0.77577 | 0.43548 | 0.09156 |

| Experiment | Change in recovery-task mean F1 | Failed retention checks | Selected update |
| --- | ---: | ---: | ---: |
| All-task continuation | -0.00361 | 14 | 0 |
| Weak-task-only continuation | -0.00262 | 11 | 0 |
| 25% replay | -0.00836 | 10 | 0 |
| Replay + L2-SP, coefficient 0.1 | -0.00836 | 10 | 0 |
| Replay + distillation, coefficient 0.1 | -0.00801 | 11 | 0 |
| Replay, learning rate 1e-5 | -0.00470 | 11 | 0 |
| Replay + PCGrad | -0.00760 | 10 | 0 |
| Replay + L2-SP, coefficient 10 | -0.00825 | 11 | 0 |

Failed-check counts are metric checks, not numbers of damaged tasks. They cover
protected-task accuracy/F1, pooled NLL/Brier and task-mean AUROC. A positive
recovery mean F1 gain of at least 0.02 was also required. No branch attains it.
The full per-task metrics, confusion matrices, AP, recalls and failed-check
names are retained in the accompanying machine-readable receipt.

## What the results show

Weak-task-only continuation provides no safe recovery: AF F1 rises slightly,
but blood-pressure, respiratory-rate and sleep-breathing F1 decline. Several
protected HRV tasks also decline. Its tiny overall macro-F1 increase is outweighed
by lower AUROC and worse probability errors. The following directly addresses
whether training only the weaker tasks hurts the others:

| Task | Parent accuracy | Weak-only accuracy | Parent F1 | Weak-only F1 |
| --- | ---: | ---: | ---: | ---: |
| activity_label | 37.16% | 38.18% | 0.2141 | 0.2221 |
| af_label | 71.64% | 71.69% | 0.4736 | 0.4768 |
| arrhythmia_category | 74.54% | 74.65% | 0.5607 | 0.5708 |
| blood_pressure_category | 67.11% | 67.48% | 0.2139 | 0.2073 |
| heart_rate_category | 87.22% | 87.33% | 0.7088 | 0.7125 |
| hrv_pnn50_category | 64.81% | 63.41% | 0.4391 | 0.4241 |
| hrv_rmssd_category | 61.54% | 60.00% | 0.4829 | 0.4722 |
| hrv_sdnn_category | 72.79% | 72.67% | 0.4481 | 0.4479 |
| rr_category | 55.94% | 55.64% | 0.3844 | 0.3807 |
| sdb_label | 50.48% | 48.73% | 0.1701 | 0.1668 |
| spo2_category | 78.94% | 78.90% | 0.2206 | 0.2205 |
| sqi_category | 88.33% | 89.49% | 0.5513 | 0.5626 |
| stress_label | 31.31% | 32.32% | 0.2060 | 0.2104 |

Replay reduces some drift relative to weak-only but does not satisfy retention.
Distillation and the smaller learning rate reduce some probability-error damage
relative to replay, while remaining worse than the parent. Normalized L2-SP at
0.1 barely changes replay; coefficient 10 still fails. PCGrad costs more updates'
computation and fails to recover the target metrics. These outcomes apply to the
tested coefficients, sample and budget, not every possible implementation of
these methods.

Three predefined parent/replay weight interpolations at 0.25, 0.5 and 0.75 also
fail. Even 0.25 gives NLL 0.76439, Brier 0.43049 and recovery mean F1 change
-0.00163. There is no promising finalist to justify a multi-seed expansion.
No fresh calibration, historical-test scoring or latency comparison was performed
for rejected candidates. Fine-tuning unchanged weights/architecture supplies no
evidence of faster inference. Prior full-run results remain in
[the full-003 report](paya-full-003.md).

## Wiring and data evidence

The native 16-example diagnostic spans all 13 tasks, matches fresh frozen
features to the bank, and learns 16/16. NLL falls from 0.84478 to 0.03242.
The saved checkpoint reloads through the public API with matching probabilities.
This is memorization/wiring verification, not a generalization benchmark.

The metadata audit covered 1,855,338 non-test QA rows and 3,490
connected components. Mapped answers agree with the supplied QA provenance;
there are no conflicting answers for an identical waveform/task in this audit.
That verifies mapping consistency, not physiological ground truth. Known
subject/recording/waveform identities stay disjoint; undisclosed aliases and
near-duplicate temporal overlap remain limitations.

The rarest severe sleep-breathing and severe-hypoxemia labels each span only
six independent groups across the available non-test pool. Many waveform rows
do not replace independent subjects. Existing selection has no severe
sleep-breathing positives and only one moderate-hypoxemia positive, limiting
the current comparisons for those labels.

Fixed-pool coverage: train: 64,000 QA rows, 52 observed task/answer pairs, 0 missing pairs; monitor: 16,384 QA rows, 50 observed task/answer pairs, 2 missing pairs.

## Draft group-stratified roles: not used for model fitting

Coverage feasibility was established and metadata-only role manifests were
prepared. They reference existing rows without copying waveform data; historical
test membership is unchanged. All 13 tasks and 52 task/answer pairs occur in each
draft role. The achieved fractions are approximately 76.98/10.38/6.60/6.03,
because independent-group class coverage takes priority over exact 80/10/5/5.

| Draft role | QA rows | Connected groups | Task/answer pairs |
| --- | ---: | ---: | ---: |
| train | 1,428,313 | 2,823 | 52 |
| selection | 192,594 | 347 | 52 |
| calibration | 122,471 | 148 | 52 |
| confirmation | 111,960 | 172 | 52 |

This is not a cure for rare independent support. Severe hypoxemia has only
8 selection, 5 calibration and 9 confirmation positives, each from one group;
severe sleep breathing has one independent group in each held-out role too.
The current parent was trained on some newly reserved groups. Any experiment
using these roles must train a fresh supervised baseline and cannot treat the
old parent as independent. Confirmation predictions remain uncomputed/locked.
No pilot above used these new roles.

## Tracking, artifacts and decision

All eight W&B runs were read back through the online API; each is finished and has loss points through update 400. Cloud model-artifact uploads are disabled; local GPU checkpoints
and recovery files remain authoritative. Only small aggregate reports were
retrieved locally. No datasets or checkpoints were downloaded or published.

- [All-task continuation](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/m7aps2oy)
- [Weak-task-only continuation](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/jd1dbchm)
- [25% replay](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/imy68vpf)
- [Replay + L2-SP, coefficient 0.1](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/wsbeen1s)
- [Replay + distillation, coefficient 0.1](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/n2bb35ak)
- [Replay, learning rate 1e-5](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/xgc4qde9)
- [Replay + PCGrad](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/gmpuyhw3)
- [Replay + L2-SP, coefficient 10](https://forge.coreweave.com/wandb/yeabsiramersha58-beijing-institute-of-technology/Paya/runs/s0skyglr)

Active experiment, audit, analysis and postprocessing tools now live in
`scripts/`; reusable model/input/checkpoint code stays in `paya/`. Shell helpers
stay in ignored `.temp/`. Input hashes and the parent checkpoint were verified
after all eight runs. At completion the GPU was idle, with 31 GB of data-disk
space available. No automatic full-run launch remains active.

Recommendation: retain the current model. The results do not support simply
continuing longer on the weak tasks, and do not establish that a new split or
encoder will improve generalization. Review the separate fresh-baseline and
encoder-comparison design before spending a larger training budget. NormWear
will be isolated with its own configuration, head and results; replacement is
the user's decision regardless of its scores. Raw multimodal data/QA preparation
is a later phase and is not mixed into these eight comparisons.

Receipt: [all metrics and audit details](paya-finetuning-001-metrics.json).
