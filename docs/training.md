# Training and run tracking

Configure assets, dataset paths, preparation budgets and optimizer settings in [config/paya.yaml](../config/paya.yaml).

## Dataset

Each JSONL row contains:

| Field | Content |
| --- | --- |
| `state` | Signal samples and metadata, using the public input format |
| `question` | One nonempty question |
| `criteria` | 2â€“64 distinct nonempty answer strings, in output order |
| `target` | Zero-based correct criterion index |
| `source_id`, `group_id`, `recording_id` | Audit identifiers for the window, subject/group and recording |

Keep all windows and questions from a recording in one split. Train, selection, calibration and test splits must have disjoint audit identifiers. Training shuffles candidate order while preserving the target mapping.

## Signal preparation

Dataset states require `sensor_type`, `sampling_rate_hz` and `duration_s` on every channel. `signal_condition` defaults to `raw`; `processed` skips NeuroKit. Use recordings appropriate to the question, with adequate duration and signal quality. Sampling rates and amplitude scales are retained. Frozen feature caches include the preparation identity, signal metadata, native samples and elapsed intervals. See [preprocessing](preprocessing.md).

## Run

```powershell
wandb login
python -m paya.train --config config/paya.yaml --run-dir runs/paya-v1 --device cpu
python -m paya.calibrate --config config/paya.yaml --checkpoint runs/paya-v1/best.pt --output runs/paya-v1/calibrated.pt
python -m paya.evaluate --config config/paya.yaml --checkpoint runs/paya-v1/calibrated.pt --split test --output runs/paya-v1/test.json
python -m paya.export --config config/paya.yaml --checkpoint runs/paya-v1/calibrated.pt --output artifacts/paya-v1
```

Raw channels are cleaned at their declared rate; processed channels bypass NeuroKit. Native variable-length windows are encoded with frozen MantisV2, and frozen BioLORD independently encodes the question and choices. Only the shared decision head is trained, using candidate cross-entropy. Every epoch is eligible for `best.pt`, selected by independent selection NLL. Calibration then fits one global temperature on separate outcomes; it is bound to the decision-head weights and invalidated by further training.

The run directory holds `history.jsonl`, resumable `last.pt` and `best.pt`. Resume with the same model, datasets and optimizer settings:

```powershell
python -m paya.train --config config/paya.yaml --run-dir runs/paya-v1 --resume runs/paya-v1/last.pt
```

## W&B

The default configuration enables online tracking:

```yaml
wandb:
  project: paya
  mode: online
  log_every: 10
```

Optional `entity` selects a team and `name` labels the run. Use `mode: offline` to save locally without authentication, or `mode: disabled` to skip tracking. Omitting the W&B section also disables tracking.

W&B records configuration, model identity, dataset counts/checksums, answer loss, gradient norm, learning rate, epoch duration, selection accuracy/NLL/Brier/ECE. Each best checkpoint produces a `model` artifact with alias `best`, containing inference weights and identity; optimizer state and raw recordings are excluded. Full resumable checkpoints stay local.

Online resume reuses the saved W&B run ID, project and entity. Offline resume creates a new local run linked by `resumed_from_run_id`; the checkpoint restores optimizer state, random generators and global step in either mode. W&B finishes the run on success or failure. Offline logs can later be uploaded with `wandb sync <offline-run-directory>`.

Reference: [W&B run API](https://docs.coreweave.com/products/wandb/ref/python/experiments/run).
