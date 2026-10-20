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

Pending: frozen-feature audit and matched small experiments. Raw operational
scripts live in ignored `.temp/`; durable methods/results are committed here.
