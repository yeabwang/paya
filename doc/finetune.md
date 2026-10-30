# Training and fine-tuning

The only trainable module is the attention decision head. SPOTR and BioLORD remain frozen. Fresh training starts with a newly initialized head. Fine-tuning starts from a compatible attention checkpoint with a fresh optimizer and temperature 1. Recovery continues the exact saved run, including its optimizer and random states.

## Dataset contract

Use JSONL files, bounded Parquet files, or JSON manifests listing Parquet/JSONL files. Each QA row contains:

```json
{
  "state": {"signals": [{"samples": [0.1, 0.2], "metadata": {
    "sensor_type": "ppg", "sampling_rate_hz": 125,
    "duration_s": 0.016, "signal_condition": "processed"
  }}]},
  "question": "What is the heart rate category?",
  "criteria": ["bradycardia", "normal", "tachycardia"],
  "target": 1,
  "task": "heart_rate_category",
  "source_id": "recording-a:window-1",
  "group_id": "subject-a",
  "recording_id": "recording-a"
}
```

The two samples illustrate the schema, not enough physiological evidence for this question. Supply suitable real recording windows and verified labels. `target` indexes that row's candidates; candidate shuffling remaps it. Audit identifiers are required for split isolation and do not enter learned scoring.

Supply a stable `task` for per-task/class reports. Every row for a task must have the same candidate vocabulary, though candidate order may vary. Missing task labels are counted explicitly; the evaluator never substitutes dataset name or guesses a task from question wording. Import task labels from original provenance when needed.

Train, selection, calibration and test groups must be disjoint by known subject, recording and waveform identifiers. Calibration is never optimized or used to select a checkpoint. Evaluate every intended signal type, rate and population with adequate class support. New candidate wording or accepted signal inputs alone do not establish generalization.

## Fresh training

Edit `config/paya.yaml` to point to your manifests and pinned assets. The default recipe uses width 128, four heads, dropout 0.1, AdamW at `3e-4`, weight decay 0.01, batch 512, gradient clipping at norm 1, and cross-entropy with 0.05 label smoothing over real candidates only. Cross-entropy consumes logits; softmax probabilities are used for prediction and reporting.

```bash
python -m paya.train --config config/paya.yaml --run-dir runs/attention --device cuda
```

The default maximum is three epochs / 9,594 updates, with full-selection NLL checks every 1,599 updates and recovery checkpoints every 100. The final update of each epoch/budget is evaluated if not already checked. Patience five counts unsuccessful selection checks. `best.pt` is the lowest-selection-NLL head; `last.pt` is recovery state. Step logs and task/class reports are saved locally. W&B can be disabled in configuration; it is not required for training.

Missing classes retain undefined recall/AUROC. Inspect macro F1, per-class support/recall and protected tasks alongside NLL, Brier and ECE. A low pooled ECE alone does not establish useful accuracy or reliable probabilities on every task.

## Fine-tune on another dataset

Copy `config/paya.yaml` or use `config/finetune.yaml`, replacing data/cache paths and setting a suitable budget. Keep the same attention dimensions and pinned encoders when loading weights. Candidate labels and task counts can change without changing the head. Mix suitable prior-domain training data if preserving previous capabilities is an objective, and evaluate those domains separately.

```bash
python -m paya.train --config config/finetune.yaml --initialize-from runs/attention/best.pt --run-dir runs/new-data --device cuda
```

`--initialize-from` copies the compatible head weights, records the parent checkpoint hash, discards its optimizer and clears its calibration. It does not resume training. Refit calibration after selecting the new weights. Calibration excludes both the new run's train/selection identities and inherited train/selection identities recorded by the parent.

## Resume an interrupted run

```bash
python -m paya.train --config config/paya.yaml --run-dir runs/attention --resume runs/attention/last.pt --device cuda
```

Recovery requires the same configuration, supported source, frozen assets and dataset content. It restores optimizer, CPU/CUDA/Python RNGs, batch cursor, selector/early-stopping state and committed logs. Do not combine `--resume` and `--initialize-from`. Use the same runtime/hardware for numerical recovery; CPU tests do not establish GPU determinism.

## Calibrate, evaluate and export

```bash
python -m paya.calibrate --config config/paya.yaml --checkpoint runs/attention/best.pt --output runs/attention/calibrated.pt --device cuda
python -m paya.evaluate --config config/paya.yaml --checkpoint runs/attention/calibrated.pt --split test --output runs/attention/test.json --device cuda
python -m paya.export --config config/paya.yaml --checkpoint runs/attention/calibrated.pt --output models/export
```

Temperature is fitted by calibration NLL on independent groups. The raw checkpoint is preserved. Reports identify the exact checkpoint, temperature, data and prediction hashes; calibration records raw/scaled pooled and task metrics. Compare both variants without choosing on test outcomes. Temperature scaling preserves answer ranking and cannot repair zero-recall classes or missing evidence. Historical test data already inspected during development remains retrospective.

Export uses an empty destination and verifies assets before copying the head, pinned encoders, supported code and checksums. Load with `paya.inference.load_model`; use the exported `code/` on `PYTHONPATH` when running independently of this checkout. Run public inference, permutation, fresh/cache, export/reload and hardware latency checks on the exact trained artifact before selecting it for use.
