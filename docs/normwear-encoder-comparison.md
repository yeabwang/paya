# Pending Paya encoder comparison

Status: source audit complete; no NormWear weights downloaded, custom code executed,
adapter implemented or training started at the end of the eight pilots. The user
subsequently authorized publishing the current model privately, creating the
branch named exactly `normwear`, preparing raw multimodal data and conducting a
separate full-training/evaluation comparison after the pilot report. NormWear remains a
separate experiment: dedicated configuration, head, run directories, checkpoint
names and reports. It must not replace, resume or alter the current main run or
become the default model. The user decides on replacement after reviewing results.

## What changes

Replace the frozen SPOTR signal encoder with frozen NormWear. Keep the public
state/metadata/question/criteria input, supplied QA targets, text encoder,
MLP fusion head structure, selection objective and reporting protocol. “Full
training” means fitting the supervised head on the full declared training role;
unfreezing the signal backbone would be an additional experiment.

The [official model card](https://huggingface.co/mosaic-laboratory/normwear)
identifies physiological pretraining and an Apache-2.0 license. The
[upstream license](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/blob/07517fcb13def8c89cb586128359cec02f86ec8d/LICENSE)
was also inspected. Preserve its license and attribution when vendoring code.
The HF repository contains both a safetensors and a pickle weight copy; obtain
only the pinned safetensors file directly on the GPU when approved.

Audited HF revision: `c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84`.
Audited official GitHub revision: `07517fcb13def8c89cb586128359cec02f86ec8d`.
The local aggregate source-checksum receipt is `.temp/reports/next/normwear-source.json`.

## Adapter requirements established by inspection

- The [configuration](https://huggingface.co/mosaic-laboratory/normwear/blob/c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84/config.json)
  specifies 768-dimensional embeddings and a 12-layer encoder. Paya currently
  uses a 512-dimensional signal vector. The head needs an explicit signal-width
  configuration and corresponding checkpoint/cache identity checks.
- The [implementation](https://huggingface.co/mosaic-laboratory/normwear/blob/c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84/normwear.py)
  computes continuous wavelet transforms of the waveform and its first two
  differences. Those three transform components are not three supplied sensors.
  Pool a declared deterministic CLS representation for our one PPG channel;
  disable reconstruction and the unrelated zero-shot text-fusion path.
- Preserve the complete 1,250 samples at 125 Hz. The HF forward path has no
  sampling-rate argument. In the inspected
  [official wrapper](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/blob/07517fcb13def8c89cb586128359cec02f86ec8d/main_model.py),
  resampling is conditional on rates above 256 Hz; 125 Hz is left unchanged.
  Do not resize the recording to the configured reconstruction length of 388.
  Dynamic positional interpolation and actual native-length shapes still need
  runtime verification before accepting this adapter.
- Repeated evaluation must return stable features, independent of training
  masking. Inspect the deterministic embedding method rather than assuming
  that every custom forward path is equivalent. Do not copy the model-card
  global integer-dtype aliasing workaround into Paya.

No runtime compatibility, memory, throughput, latency or accuracy claim follows
from this static inspection. Longer token sequences and CWT preparation can
increase cost; measure them on the rented GPU before committing a full budget.

## Controlled experiment after review

1. Use one declared protocol for both encoders. If the new group-stratified roles
   are adopted, train both supervised heads from scratch. The current parent was
   trained on some groups now reserved for calibration/confirmation and cannot
   be used as an independent starting point for those roles.
2. Verify finite deterministic features on actual native inputs, frozen-backbone
   behavior, 16-example fitting, public API probability agreement and checkpoint
   reload. Compare cached and freshly computed features. Use a distinct cache
   identity; never reuse SPOTR embeddings as NormWear features.
3. Run a matched small screen. Keep text assets, head hidden width, dropout,
   smoothing, loss, sampling, optimizer settings and update/epoch budget fixed.
   Account for the extra trainable parameters in the wider input projection;
   do not call the two heads parameter-identical. If encoding batch size must
   shrink, keep the effective supervised training batch comparable.
4. After review of the small screen and measured GPU budget, perform the requested
   full NormWear training as the final encoder comparison. Select by the same
   selection rule, calibrate on the independent calibration role, and compare
   raw/calibrated accuracy, task macro F1/AUROC/AP, NLL, Brier and ECE. Use
   connected-group uncertainty estimates and disclose rare-class support.
5. Unlock confirmation only for the predeclared finalist. Historical test remains
   a retrospective benchmark because prior cycles have already exposed its
   results. Measure synchronized batch-one public-API p50/p95/p99 latency with
   CWT and signal encoding included, signal-feature caching disabled, and both
   warm and cold text conditions reported. Keep both models and present the
   evidence to the user; no metric threshold authorizes automatic replacement.

Pretraining overlap with public cohorts must be investigated and disclosed;
changing the encoder does not establish an unseen-cohort evaluation. A single
matched seed is a first comparison, not proof of a reproducible encoder benefit.

## Additional raw multimodal experiment requested by the user

After receiving the previous-experiment report, the user wants WESAD, UCI HAR
and other datasets specifically used for NormWear pretraining, with the current
PPG dataset available as an additional source. UCI HAR was explicitly confirmed.
No downloads or dataset preparation for this phase have started.

First verify the exact pretraining inventory, licenses/access terms, raw versus
already processed releases, modalities and reference labels. Generate questions
and candidate choices only for outcomes supported by released labels. Keep QA
paraphrases for a recording in the same subject/recording split and preserve
provenance. Do not invent clinical labels from unrelated activity or stress labels.

This is a distinct comparison from replacing only the encoder on current data.
Use the same recordings, tasks, QA, candidate vocabulary and split roles for
matched encoder comparisons; train fresh supervised heads with encoder-appropriate
feature dimensions and preprocessing. Report parameter counts and any unavoidable
preprocessing differences. A multimodal NormWear head trained on new data cannot
be credited with an encoder-only gain over the historical PPG-only run.

UCI HAR contains inertial activity signals. Current Paya input validation and
cleaning explicitly support PPG, ECG, EEG and iEEG, not inertial sensor types.
Any inertial adapter and preprocessing policy must be isolated and verified;
do not route those channels through physiological cleaning or imply existing
support. Compare both encoders on supported shared modalities, and label any
NormWear-only modality results separately. Pretraining-cohort reuse must be
disclosed; supervised subject separation does not prove pretraining independence.
