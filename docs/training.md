# Training and run tracking

Configure assets, dataset paths, preparation budgets and optimizer settings in [config/paya.yaml](../config/paya.yaml).

[config/paya-full.yaml](../config/paya-full.yaml) defines the first full-data Paya baseline: three epochs, batch size 16, a fresh decision head, online metrics and local model checkpoints. It uses the processed split manifests under `data/training`, preserves the inherited splits, selects `best.pt` by selection NLL, and evaluates that checkpoint on the test split. No independent calibration split is available for this run, so report uncalibrated results with temperature 1. Rare or absent validation classes remain a coverage limitation; inspect per-task results alongside pooled metrics.

## Dataset

JSONL rows and Parquet rows share the same input format:

| Field | Content |
| --- | --- |
| `state` | Signal samples and metadata, using the public input format |
| `question` | One nonempty question |
| `criteria` | 2-64 distinct nonempty answer strings, in output order |
| `target` | Zero-based correct criterion index |
| `source_id`, `group_id`, `recording_id` | Audit identifiers for the window, subject/group and recording |

Keep all windows and questions from a recording in one split. Train, selection, calibration and test splits must have disjoint audit identifiers. Training shuffles candidate order while preserving the target mapping.

Dataset paths accept one JSONL/Parquet file, a directory of Parquet shards, or a small JSON manifest containing `files` and an optional `path_base`. Each configured split needs its own path or manifest. Rows retain the same state, question, criteria, target and audit fields. No modality is added to the model name.

The loader hashes the inputs and builds a disposable SQLite metadata index alongside the dataset. Waveforms remain in their source files. SQLite performs split overlap checks on disk. Iteration holds one block plus the configured batch, and shuffles blocks and rows deterministically. Parquet groups above 2,048 rows or 64 MiB of encoded uncompressed data or decoded numeric values are rejected; write groups of 256 rows for large datasets. Counting numeric values also guards against dictionary encoding hiding large repeated signals. JSONL blocks contain at most 256 rows and 64 MiB of source text. Python objects and model memory add overhead beyond those input-byte limits. `dataset.rows` is a lazy sequence; explicitly converting it to a list still materializes the requested data.

Training and CLI evaluation/calibration write predictions to JSONL and aggregate accuracy, NLL, Brier and ECE with constant-size accumulators. Evaluation reports reference their sibling `.predictions.jsonl` file rather than embedding all predictions. Library `collect` also returns a reusable disk-backed `PredictionRecords` iterable; without an output path it owns a temporary file. Calibration provenance references content-checked training/selection datasets instead of embedding millions of audit IDs.

## Signal preparation

Dataset states require `sensor_type`, `sampling_rate_hz` and `duration_s` on every channel. `signal_condition` defaults to `raw`; `processed` skips NeuroKit. Use recordings appropriate to the question, with adequate duration and signal quality. Sampling rates and amplitude scales are retained during cleaning. Frozen feature caches store only the signal embedding and checksums, bound to the preparation, native signal and encoder identities. See [preprocessing](preprocessing.md).

## Run

```powershell
wandb login
python -m paya.train --config config/paya.yaml --run-dir runs/paya-v1 --device cpu
python -m paya.calibrate --config config/paya.yaml --checkpoint runs/paya-v1/best.pt --output runs/paya-v1/calibrated.pt
python -m paya.evaluate --config config/paya.yaml --checkpoint runs/paya-v1/calibrated.pt --split test --output runs/paya-v1/test.json
python -m paya.export --config config/paya.yaml --checkpoint runs/paya-v1/calibrated.pt --output artifacts/paya-v1
```

Raw channels are cleaned at their declared rate; processed channels bypass NeuroKit. Frozen SPOTR encodes each complete recording once in supplied channel order and returns one 512-dimensional global token. Supported types are EEG, iEEG, ECG and PPG; channels within a recording must share rate, length and start time. Frozen BioLORD independently encodes the question and choices. Only the shared decision head is trained, using candidate cross-entropy. Every epoch is eligible for `best.pt`, selected by independent selection NLL. Calibration then fits one global temperature on separate outcomes; it is bound to the decision-head weights and invalidated by further training.

The run directory holds `history.jsonl`, resumable `last.pt`, `best.pt` and streamed selection predictions. Checkpoints, cache entries and prediction files use flushed, synced temporary files and atomic replacement. A failed write preserves the previous committed file. Configure `optim.checkpoint_every_steps` (default 100) to bound replay after a crash; epoch ends also checkpoint. A hard crash can lose work since the last committed checkpoint. Resume with the same model, datasets and optimizer settings:

```powershell
python -m paya.train --config config/paya.yaml --run-dir runs/paya-v1 --resume runs/paya-v1/last.pt
```

Resume restores the next batch within the epoch, accumulated epoch loss, optimizer, RNG state, completed epoch history and W&B identity. Shuffling does not depend on how many earlier blocks were decoded, so it can skip committed batches without replaying their waveform reads. Dataset content hashes must match, and changes detected during iteration abort the run. Local history is reconciled to the committed checkpoint. Online tracking records the recovered step and may replay its final saved metrics; remote logging is not a filesystem transaction. Keep the dataset files/manifests available for later calibration provenance checks. Do not edit a dataset during an active run.

## W&B

The default configuration enables online tracking:

```yaml
wandb:
  project: paya
  mode: online
  log_every: 10
```

Optional `entity` selects a team and `name` labels the run. Use `mode: offline` to save locally without authentication, or `mode: disabled` to skip tracking. Omitting the W&B section also disables tracking.

`wandb.log_models: false` keeps live metrics and local resumable checkpoints while skipping cloud model artifacts. Its default is `true`.

W&B records configuration, model identity, dataset counts/checksums, answer loss, gradient norm, learning rate, epoch duration, selection accuracy/NLL/Brier/ECE. Each best checkpoint produces a `model` artifact with alias `best`, containing inference weights and identity; optimizer state and raw recordings are excluded. Full resumable checkpoints stay local.

Online resume reuses the saved W&B run ID, canonical project name and entity. Offline resume creates a new local run linked by `resumed_from_run_id`; the checkpoint restores optimizer state, random generators and global step in either mode. A durable `steps.jsonl` records each optimizer step. Before a checkpoint is published, this log is flushed and synced and its byte offset is stored in the checkpoint. Resume truncates uncommitted log tails and replays logged metrics to the same W&B run with `recovery/replayed: true`; repeated logical steps have the same recorded values. Keep this log beside the resumable checkpoints.

Tracking initializes within 90 seconds and final synchronization has a 120-second timeout by default (`wandb.finish_timeout` can override it). A synchronization timeout raises an error and preserves the local model/checkpoints and W&B event files for retry, rather than reporting a successful cloud sync. Offline or interrupted event files can be uploaded with `wandb sync <run-directory>`. Service outages can delay live metrics and artifacts even though local training work is committed.

Reference: [W&B run API](https://docs.coreweave.com/products/wandb/ref/python/experiments/run).
