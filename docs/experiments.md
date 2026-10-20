# Paya improvement experiments

## Protocol (2026-10-21)

Reference: `paya-full-002`, code `67d6d61`. All three epochs completed;
epoch 1 was selected by selection NLL. Nine tasks had zero minority-class
recall. Reported test accuracy alone is insufficient.

The existing train, selection, independent calibration and test memberships
remain fixed for this improvement cycle. Development uses train and selection
only. The already inspected test is a retrospective reference, not an unseen
benchmark; do not use it to select hyperparameters. Any future replacement split
must group connected subjects/recordings/waveforms and report a separate protocol.

Experiments must record code/config/data hashes, seed, training size, candidate
vocabulary, optimizer budget, class coverage, elapsed time, checkpoint selection,
and online W&B identity. Compare on identical selection rows. Report per-task
accuracy, macro F1, supported-class one-vs-rest AUROC, NLL, Brier and ECE. Undefined
AUROCs are null with support counts, not zero. Include waveform ablations and
class recall. Keep all small failures in the ledger, not only winners.

Start with a wiring/feature audit, then a balanced small fit. Expand only promising
changes to a larger, representative selection evaluation. Treat raw accuracy,
macro F1, probability quality and latency as separate objectives: report tradeoffs
rather than promising all metrics improve together. Preserve the completed run
and its checkpoints. No publication or instance changes are part of this work.

## Research and hypotheses

- [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314)
  motivates controlled class-prior/loss experiments. Reweighting changes the
  effective prior and can harm probability metrics; evaluate both ranking and
  calibration on natural-prevalence selection data.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
  motivates separate temperature fitting. A single fitted temperature is not
  evidence of transfer under task/population shift, as the reference run showed.
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)
  motivates the 0.05 smoothing ablation after the tiny fit showed extreme
  held-out overconfidence. Results on other domains are hypotheses here.
- [Calibrating Deep Neural Networks using Focal Loss](https://arxiv.org/abs/2002.09437)
  is a possible later comparison if initial regularization fails; focal loss is
  not assumed to improve this signal corpus without a matched experiment.
- [scikit-learn AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  and [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  define the ranking and class-sensitive checks. Align probabilities by answer
  text; candidate positions are shuffled during training.

Initial hypotheses, not established causes:

1. Frozen signal embeddings have a large shared component or weak native-rate
   information relative to text; the head may ignore waveform variation.
2. One signal token makes cross-attention weights identically one. Query/key
   projections cannot learn signal selection, although downstream nonlinear
   layers can still use the broadcast signal value.
3. Long-tailed labels and correlated block batches favor majority-class answers.
4. Normalization/direct signal-text interactions or richer signal tokens could
   improve expressiveness. Test the cheaper head-only interventions before
   changing frozen encoders or invalidating feature caches.

## Ledger

### Audit and balanced memorization

The 309-row integration corpus had 263 distinct frozen FP32 feature rows.
Mean feature norm 12.90, mean-vector norm 8.38, mean centered norm 9.81.
Cached and fresh batched features matched within the established tolerance.
The completed checkpoint scored 29.70% on the 101 balanced selection examples;
mean/zero replacement scored 25.74%, shuffled features 23.76%. This small,
deliberately balanced diagnostic has different prevalence from full selection;
it establishes signal sensitivity, not a generalization result. Duplicate
features alone do not establish corruption because QA can share a waveform.

Three seed-42, 400-update experiments used exactly 208 train/101 selection rows,
batch 64, AdamW lr 0.0003/weight decay 0.01. All used frozen cached encoders and
online W&B. The original head reached 100% train accuracy by step 400; adding
signal LayerNorm also reached 100%. A 128-wide direct interaction MLP reached
100% by step 100. Generalization did not follow: late selection NLLs exceeded
5, while selection accuracy stayed around 28-38%. Do not promote these highly
overfit heads. The architecture can memorize the supplied small multi-task
labels; disconnected gradients or a universal inability to represent them are
not supported explanations for the full-run failure.

Next: a deterministic label-independent 1/16 QA sample of native train, with
the whole unchanged selection corpus available for expansion. First compare
8192 sampled training examples and a fixed 4096-row selection sample. Test
natural cross-entropy, clipped square-root class weighting, label smoothing,
and the smaller interaction head. All reported comparisons use the same rows
and optimizer budget. Expand successful candidates only. Small pilots may miss
rare classes; report support and verify on full selection before conclusions.

### Preparation and reproducibility

The development bank stores FP32 frozen features plus independently indexed
text vectors, candidate masks, targets and task labels used only for reporting
and training weights. It validates mapping-ID alignment during extraction and
checksums all bank files before use. Partial banks have no completion manifest
and cannot train. It does not load calibration or test. Sampling uses a fixed
mapping-ID hash and is independent of labels. The bank retains only the existing
native train/selection memberships and pins the original frozen model identity.

On 20 representative row groups (320 sampled rows), converting complete Arrow
groups to Python took 3.063 seconds; selecting rows in Arrow first took 0.578
seconds. The builder now selects before conversion and reuses file readers.
This is a preparation microbenchmark, not a measured end-to-end training speedup.

Experimental training checkpoints include optimizer, RNG, online run ID, source
hashes, exact plan and bank bindings. They resume the same bounded experiment;
completed variants are not relaunched. Pilot draws are uniform with replacement
from the fixed sample, batch 256, 800 optimizer updates, evaluation every 200.
Head selection uses minimum selection NLL; macro F1/AUROC and waveform ablation
remain separately visible. Weighted variants use training-only square-root
inverse class frequency, raw cap 4, then task-wise normalization to unit mean.
Smoothing is 0.05 over real candidates only, never padding.

Verification before pilot: 158 local unit tests passed, followed by a new bank
alignment/corruption check and the existing focused checks (9 passed). The
experimental interaction head is not wired into the public model or promoted.

### Pilot outcome and expansion decision

Six 8,192-train/4,096-selection pilots completed; all use the same rows, seed,
800 updates and the predeclared lowest-NLL checkpoint selector. Reference
`paya-full-002` on these selection rows: accuracy 68.29%, NLL 0.97111, Brier
0.47526, ECE 0.08406, mean task macro F1 0.26073, mean task macro AUROC 0.67613.

| Pilot | Accuracy | NLL | Brier | ECE | Task macro F1 | Task macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention CE | 65.58% | 0.91071 | 0.50733 | 0.07926 | 0.21266 | 0.56617 |
| Attention weighted | 62.62% | 0.90941 | 0.51949 | 0.06759 | 0.24487 | 0.63104 |
| Attention smoothing | 65.60% | 0.89056 | 0.50305 | 0.05969 | 0.21418 | 0.55691 |
| Attention weighted + smoothing | 62.94% | 0.89861 | 0.51677 | 0.06493 | 0.24166 | 0.62478 |
| Interaction CE | 65.06% | 0.84230 | 0.47643 | 0.06087 | 0.27150 | 0.61407 |
| Interaction weighted + smoothing | 61.08% | 0.88825 | 0.50963 | 0.02633 | 0.30490 | 0.60730 |

These are tradeoffs, not an overall improvement claim. The CE attention head's
lowest-NLL checkpoint has identical accuracy and F1 under mean-signal ablation;
later checkpoints use signals more but score worse on NLL. This exposes a real
selection-objective conflict under imbalance. Late weighted/smoothed attention
reaches F1 0.32190 but NLL 0.94352. Do not report that F1 as the selected model's
score or choose it from test results.

Expand the unchanged attention control, weighted/smoothed attention, and CE
interaction head using all 102,345 development training rows, batch 512 and
1,600 updates. A fixed 16,384-row selection sample chooses checkpoints every
400 steps; then evaluate those checkpoints once on all 196,707 selection rows
and under a mean-signal ablation. The whole selection set remains development
data, so these comparisons do not constitute an untouched test result. No pilot
variant has been promoted to the public inference model.

### Expanded results (all 196,707 selection rows)

| Model | Accuracy | NLL | Brier | ECE | Task macro F1 | Task macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-run reference | 67.12% | 1.01766 | 0.49039 | 0.09868 | 0.26318 | 0.63589 |
| Attention CE | 68.37% | 0.80803 | 0.45195 | 0.10195 | 0.33156 | 0.68711 |
| Attention weighted + smoothing | 66.88% | 0.78627 | 0.44779 | 0.03186 | 0.36954 | 0.68515 |
| Interaction CE | 68.43% | 0.81303 | 0.45140 | 0.09089 | 0.35053 | 0.70535 |

The 264,065-parameter interaction head improves all six displayed validation
metrics over the 2,567,681-parameter reference head. It was selected at step
800 by the declared 16,384-row subset NLL, not by test results. This is evidence
to expand, not a test generalization claim. Training used 102,345 rows rather
than the reference run's 1,637,289; optimization/data budgets differ and are
explicit. No prior full-run hyperparameters were changed retroactively.

Waveform sensitivity is substantially stronger in the larger runs: mean-signal
replacement drops attention CE accuracy from 68.37% to 44.78%, and weighted/
smoothed accuracy from 66.88% to 35.42%. This rejects an explanation based only
on task label frequencies for those trained heads.

Next recursion: compare smoothing alone and weighting plus smoothing on the
smaller interaction head, with exactly the same larger-stage rows and update
budget. Then nominate a configuration before independent calibration and any
retrospective test evaluation. Baseline test results already influenced the
overall research direction; that historical test must not be called unseen.

An inference-only single-token attention shortcut skips query/key projections:
softmax over one unmasked key is exactly one. Masked values still contribute
zero. Training/dropout and multi-token attention retain the existing path.
161 unit tests passed, including explicit comparison against general attention.
Latency improvement remains to be measured on an otherwise idle GPU.

### Refinement nomination and latency

| Compact variant | Accuracy | NLL | Brier | ECE | Task macro F1 | Task macro AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE + 0.05 smoothing | 68.40% | 0.78821 | 0.44595 | 0.06921 | 0.35013 | 0.70003 |
| Weighted + 0.05 smoothing | 65.83% | 0.78893 | 0.44999 | 0.03488 | 0.35584 | 0.69633 |

Nominate smoothing without class weighting. It improves every displayed
selection metric relative to the reference, whereas weighting costs accuracy.
The task-level report still includes failures: zero recall remains for some
activity, BP, breathing, oxygen and HRV labels. AF no longer has zero recall.
These aggregate gains do not mean every task or class improved.

An idle-GPU, interleaved benchmark used 260 batch-one public API requests per
case across the same 13 selection examples/task types. Native preparation and
signal encoding are included; text embeddings are warm; no signal-feature
cache is used; CUDA is synchronized. Process/model startup is excluded.

| Case | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: |
| Reference, original attention path | 13.760 | 16.526 | 18.332 |
| Reference, equivalent attention shortcut | 13.419 | 13.513 | 16.562 |
| Compact smoothed head | 12.047 | 12.338 | 16.213 |

Baseline probabilities were identical under the shortcut on the checked
requests. The compact public checkpoint saved/reloaded and reproduced its
prediction. Median public inference latency is about 12.5% lower than the
same-session reference. This is a small idle-device measurement, not a service
load or cold-start guarantee.

### Full-data expansion

The nominated compact head is an optional `kind: interaction` configuration;
the default attention model and old checkpoints remain loadable. The public
state/question/criteria arguments are unchanged. Only head weights train.
Frozen SQLite features now bind encoders, preparation, signal representation
and cache contract, excluding head settings which cannot affect those features.
Existing stores are validated under that same rule; changed assets still fail.

The next run is `paya-full-003`: all 1,637,289 native training rows, three complete
epochs, batch 512, 3,198 steps/epoch, 9,594 total updates, AdamW lr 0.0003 and
0.05 real-candidate label smoothing. No class weighting. Deterministic
no-replacement sampling includes each QA row exactly once per epoch. Full
selection NLL chooses the checkpoint at 1,599-step intervals; no test selection.

Feature preparation journals checksummed, synced 10,000-row chunks and resumes
only if dataset/model/builder bindings match. Training checkpoints preserve
optimizer, random states, logical step, best weights and online run identity.
GPU allocation is checked against an explicit reserved budget before loading
the bank. NumPy-backed Arrow sample views avoid Python float expansion while
preserving exact preparation identities; heterogeneous structures fall back
to the existing conversion. No additional raw dataset copy is created.

Verification: 166 unit tests passed, including Arrow-view identity/fallback,
chunk corruption/recovery, public compact checkpoint reload and exact epoch
coverage. Final full-run metrics, independent
calibration and retrospective test comparison remain pending.

Full feature preparation completed in 352.81 seconds (all 1,637,289 rows).
The bank contains frozen features and small text/audit indices, not another raw
waveform copy. The full sampler and public pipeline passed 167 unit tests.

Tracking audit: early development W&B runs included generic base-config head/
batch metadata as well as the explicit bound experimental plan. The plan,
saved weights, parameter counts and result files describe the actual experiment;
do not interpret that generic metadata as its architecture or update size.
The full runner now records the actual selected head, batch, smoothing and
sampling mode directly. This instrumentation correction does not change any
previous result or prediction.

### Data observability questions for later cycles

Label agreement with the supplied QA is a data-integrity check, not independent
verification of physiological ground truth. Some targets may require information
the current normalized single-channel ten-second waveform does not preserve.
For example, [primary pulse-oximetry research](https://pmc.ncbi.nlm.nih.gov/articles/PMC4029673/)
derives saturation using two optical wavelengths and a ratio of their pulsatile/
baseline components. The present one-channel input cannot supply that two-channel
measurement; a learned category prediction can instead use correlations that
may fail across populations. This is an observability hypothesis, not a finding
that the supplied labels are wrong or a reason to relabel them.

[A 3,387-adult HRV study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4586373/)
compared ultra-short interval-based estimates with 240-300-second references
(its methods derived intervals from finger arterial-pressure recordings). Its findings
depend on the HRV statistic and recording protocol and do not validate these
PPG QA categories. Later data research should distinguish short-window label
prediction from recovery of a longer-duration physiological measurement. No
waveforms or QA labels were changed for these reasons in this cycle.

### Post-run recovery hardening

After full-003 completed, a code audit tightened future experiment bindings:
they now include the effective base configuration as well as plan, source
hashes and bank hashes. The runner checks the bank against current frozen
assets/preparation and actual source manifests, rejects mismatched completed
results, and checks GPU allocation before loading metadata/index arrays.
This prevents silently reusing a bank or result after a configuration change.
168 local unit tests passed. The completed full-003 model and its metrics still
belong to the immutable `48754f3` training/evaluation revision; these checks do
not change its weights or predictions.
