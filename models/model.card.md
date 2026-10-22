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

Paya scores candidate answers to a question about a physiological recording. Frozen SPOTR supplies global, channel and temporal signal tokens; frozen BioLORD supplies separate normalized question/candidate vectors. The trainable **480,257-parameter attention head** applies question/candidate self-attention, signal cross-attention and a shared candidate scorer. See [architecture](../doc/architecture.md) for exact operations and masks.

## Artifact status

The architecture is selected for clean training. No trained attention checkpoint is supplied in this directory yet. Screening weights are not promoted. No benchmark, calibration or latency value is claimed for the forthcoming artifact. Previous weights and matching results are archived under ignored `retired/mlp-fusion/`.

## Intended use and limits

Research on candidate-based physiological signal decisions. PPG, ECG, EEG and iEEG processing/validation is implemented, with native rates and processed-sample passthrough. Signal type/rate guide processing only; physical sensor metadata is not learned. Accepted input/candidate formats do not establish accuracy on new datasets or signals. Use verified labels, disjoint selection/calibration/test groups and adequate class coverage for every intended population.

Only the head is trained or fine-tuned. Training uses masked logits with cross-entropy and configurable label smoothing. Independent temperature calibration is cleared when weights are fine-tuned. Neither attention nor temperature scaling guarantees accuracy or low ECE; report the actual matching artifact's pooled, task and class results.

## Use after training

Follow [training and fine-tuning](../doc/finetune.md), then export the selected checkpoint and load with `paya.inference.load_model`. Export bundles contain frozen encoder assets, the head, configuration, supported code and checksums. SPOTR and BioLORD retain their upstream source attribution and licenses. The package version is unchanged.
