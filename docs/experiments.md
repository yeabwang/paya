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
