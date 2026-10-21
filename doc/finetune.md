# Fine-tuning Paya

This guide explains how to adapt the pretrained Paya decision head to your own labeled physiological recordings. It covers data preparation, initialization from the released weights, training, recovery, calibration and export. The signal and text encoders remain frozen; the trainable component is the **264,065-parameter MLP fusion head with Hadamard product features** described in [Architecture](architecture.md).

Run the examples from the repository root. Use a new run directory and preserve the downloaded release. The settings below are an illustrative starting configuration, not a demonstrated optimum for a new dataset. Historical experiment outcomes are in [Experiments](experiments.md#checkpoint-continuation-experiments).

## 1. Install the model and runtime

Use Python 3.12 and install:

```bash
python -m pip install -r requirements-model.txt
python -m pip install -e .
```

Download the complete private [Paya model bundle](https://huggingface.co/yeabwang/Paya) into `models/export/` using an authorized account. It must contain `decision.pt`, the `spotr/` and `text/` asset directories, `config.json` and `manifest.json`. Use the pinned assets and preparation versions: checkpoint loading validates their identities. The [runtime record](../models/runtime.json) lists the verified package versions.

Fine-tuning the head does not require enabling encoder gradients. Keep the released head dimensions and preparation settings when loading its weights. Changing encoders or head dimensions is a different training configuration and is not compatible with that checkpoint.

## 2. Prepare labeled examples

Store one QA example per JSONL row, or use Parquet with the same fields. This Python example constructs a valid row from an actual processed, single-channel, 10-second PPG recording:

```python
row = {
    "state": {"signals": [{
        "samples": ppg_samples,  # A list of 1,250 finite measurements.
        "metadata": {
            "sensor_type": "ppg",
            "sampling_rate_hz": 125,
            "duration_s": 10,
            "signal_condition": "processed",
        },
    }]},
    "question": "What is the heart rate category?",
    "criteria": ["bradycardia", "normal", "tachycardia"],
    "target": 1,  # Zero-based index: the correct answer here is "normal".
    "source_id": "cohort-a:waveform-0001",
    "group_id": "cohort-a:subject-001",
    "recording_id": "cohort-a:recording-001",
    "mapping_id": "example-0001",  # Optional row identifier.
    "datasetname": "cohort-a",    # Optional provenance label.
}
```

Replace the example's target with the real label. If you reorder candidates, update the target index accordingly. Keep a consistent answer vocabulary for each task when computing task-level F1 and AUROC. The loader also shuffles candidate order during training while preserving target alignment.

`source_id`, `group_id` and `recording_id` are required audit strings, not model features. Reuse the same identifier whenever rows share the corresponding waveform, subject or recording. Different questions about the same waveform must not receive different waveform identities. Namespace identifiers across independent cohorts while retaining genuine cross-dataset connections. Optional row identifiers cannot replace these grouping fields.

Set `signal_condition="processed"` only for already processed waveforms. Raw recordings use `"raw"` and pass through native-rate cleaning. Metadata must match the actual rate, duration and channel alignment; the pipeline does not resample or silently repair invalid values. See the [input contract](architecture.md#public-input-and-output) for channel, sample and text limits. New sensor modalities require their own validation; the released performance evidence covers PPG.

For Parquet, keep row groups within the loader's bounds of 2,048 rows and 64 MiB estimated uncompressed/numeric size; 256-row groups are a practical packing choice. Avoid building one enormous JSON array: `ChoiceDataset` reads bounded JSONL blocks or Parquet row groups.

## 3. Create independent data roles

Prepare four disjoint files, for example:

```text
data/custom/train.jsonl
data/custom/selection.jsonl
data/custom/calibration.jsonl
data/custom/test.jsonl
```

Group connected subjects, recordings and shared waveforms before assigning roles. Training updates weights; selection chooses the checkpoint and training settings; calibration fits the final temperature; test measures the finalized procedure. The same subject must not appear in multiple roles, even under different recordings or questions. Check class coverage and support in every role before training.

The supplied `assert_disjoint` check detects shared audit identifiers across these files. It cannot detect a shared subject or duplicate waveform that you have given different identifiers. Also account for the pretrained model's original data exposure: partitioning a reused cohort again does not make it unseen to the parent model. Preserve established held-out memberships where applicable and describe prior exposure in your evaluation report.

Start with a small fitting diagnostic and a small, representative development subset. Verify target alignment, finite loss, nonzero head gradients, frozen encoders and checkpoint reload before expanding. Fitting 10–20 examples verifies learning and integration; it is not a held-out performance estimate.

## 4. Write a fine-tuning configuration

Create `.temp/` for local configuration and helper files, then save this example as `.temp/finetune.yaml`. Paths are relative to the working directory, so run commands from the repository root.

```yaml
seed: 42
threads: 8
assets:
  spotr: models/export/spotr
  text: models/export/text
preparation:
  max_channels: 128
  max_samples: 12800
head:
  kind: interaction  # Saved compatibility identifier for MLPFusionHead.
  width: 128
  dropout: 0.1
data:
  train: data/custom/train.jsonl
  selection: data/custom/selection.jsonl
  calibration: data/custom/calibration.jsonl
  test: data/custom/test.jsonl
feature_cache: cache/custom/features.sqlite3
feature_cache_format: sqlite
batch_size: 64
optim:
  epochs: 3
  lr: 0.00003
  weight_decay: 0.01
  label_smoothing: 0.05
  checkpoint_every_steps: 100
wandb:
  project: Paya
  name: paya-custom-001
  mode: online
  log_every: 20
  log_models: false
  finish_timeout: 20
```

Configure W&B authentication in your environment before using online tracking. Its metric tracking is independent of cloud model-artifact uploads; `log_models: false` keeps checkpoints local. For a local diagnostic without tracking, set `mode: disabled`.

Choose batch size based on measured memory use and throughput. The cache stores frozen signal features, and its identity checks prevent reuse under incompatible assets or preparation. A different head learning rate does not invalidate frozen features. Use selection data to compare learning rate, smoothing and training budget; do not adjust them based on test results.

## 5. Initialize from pretrained weights and train

The current `paya.train` CLI has **no pretrained-initialization flag**. Without `--resume`, it initializes a new head. Its `--resume` option normally restores the same run's optimizer, data bindings and RNG state; passing the released checkpoint directly is not a supported way to change datasets or learning rate.

To start a new fine-tune, the following example loads the pretrained head, creates a **fresh AdamW optimizer**, evaluates the parent on your selection data, and writes an initial training checkpoint compatible with the existing trainer. Save it as `.temp/start_finetune.py`. The original release files are read only.

```python
import gc
import json
import random
from pathlib import Path

import torch
import yaml

from paya.checkpoint import save_checkpoint
from paya.data import ChoiceDataset, assert_disjoint
from paya.evaluate import collect, metrics
from paya.features import open_feature_cache
from paya.inference import load_model
from paya.provenance import sha256
from paya.train import train

config_path = Path(".temp/finetune.yaml")
run_dir = Path("runs/paya-custom-001")
device = "cuda" if torch.cuda.is_available() else "cpu"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])
torch.set_num_threads(config["threads"])

# Use a fresh directory; never overwrite a previous run.
run_dir.mkdir(parents=True, exist_ok=False)
datasets = {name: ChoiceDataset(path) for name, path in config["data"].items()}
assert_disjoint(*datasets.values())

model = load_model("models/export", device=device)
parent_path = Path("models/export/decision.pt")
assert model.head.config == config["head"]
assert model.preprocessor.identity()["config"] == config["preparation"]
model.train()  # Reset any old temperature/calibration before new optimization.
cache = open_feature_cache(config)
parent_selection = metrics(collect(
    model, datasets["selection"], config["batch_size"], cache,
    output_path=run_dir / "parent-selection.predictions.jsonl",
))
optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=config["optim"]["lr"],
    weight_decay=config["optim"]["weight_decay"],
)

state = {
    "config": config, "epoch": 0, "global_step": 0,
    "next_batch": 0, "metrics_bytes": 0,
    "best_nll": parent_selection["nll"],
    "dataset_sha256": {name: data.sha256 for name, data in datasets.items()},
    "optimizer": optimizer.state_dict(), "wandb": None,
    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    "forbidden_calibration": {"datasets": [
        {"path": str(datasets[name].path), "sha256": datasets[name].sha256}
        for name in ("train", "selection")
    ]},
}
(run_dir / "steps.jsonl").touch()
save_checkpoint(run_dir / "initial.pt", model, state)
save_checkpoint(run_dir / "best.pt", model, state)
(run_dir / "parent.json").write_text(json.dumps({
    "checkpoint_sha256": sha256(parent_path),
    "selection": parent_selection,
}, indent=2), encoding="utf-8")
# Release this model before the trainer constructs its own instance.
del model, optimizer, cache, datasets
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
train(config_path, run_dir, device=device, resume=run_dir / "initial.pt")
```

Launch the example with:

```bash
python .temp/start_finetune.py
```

The initial `best.pt` preserves the parent as update zero. If no completed epoch improves selection NLL, it remains the best checkpoint. The standard trainer updates only the head, clips gradient norm to 1, uses the configured label-smoothed loss and evaluates selection at each epoch boundary. It saves periodic recovery checkpoints every configured number of updates. **Checkpoint-save frequency does not change validation frequency.**

The current trainer does not implement a learning-rate scheduler, automatic early stopping, class weighting, task replay, distillation or PCGrad. Do not add YAML keys expecting those features to activate. Set a bounded epoch budget and compare saved selection results; such additional methods require a separately implemented and validated training loop.

## 6. Monitor and resume

The run directory contains:

| File | Purpose |
| --- | --- |
| `parent.json` | Parent checkpoint identity and selection baseline |
| `initial.pt` | Pretrained head with fresh optimizer and initial recovery state |
| `best.pt` | Lowest selection NLL, including the parent baseline |
| `last.pt` | Most recently committed recovery state, once the trainer saves one |
| `steps.jsonl` | Batch loss, gradient norm and learning rate |
| `history.jsonl` | Completed-epoch selection metrics |
| `selection-predictions.jsonl` | Most recent epoch's selection logits and audit identifiers |

Batch loss changes with task composition and is not validation performance. Compare the parent and fine-tuned candidate on identical selection examples, including per-task accuracy, macro F1, supported-class AUROC and class recall. Define any tolerated regression on other tasks before selecting a replacement. Lower pooled NLL alone does not guarantee better minority recall.

After an interruption, ensure the previous worker is stopped, then resume from the last committed checkpoint:

```bash
python -m paya.train --config .temp/finetune.yaml --run-dir runs/paya-custom-001 --device cuda --resume runs/paya-custom-001/last.pt
```

Use `--device cpu` on a CPU installation. If training stopped before creating `last.pt`, resume from `initial.pt` instead; this repeats work since initialization. Keep `steps.jsonl` beside the checkpoint. Resume verifies configuration and data hashes, restores optimizer/RNG state and discards log entries beyond the committed cursor. Do not rerun the initializer into an existing directory or change the dataset/configuration while resuming; a new experiment needs a fresh run.

## 7. Calibrate and evaluate the selected checkpoint

Once model selection is complete, optionally fit a global temperature using only the independent calibration role:

```bash
python -m paya.calibrate --config .temp/finetune.yaml --checkpoint runs/paya-custom-001/best.pt --output runs/paya-custom-001/calibrated.pt --device cuda
```

Calibration checks its identifiers against the fine-tune's training and selection bindings. Those automatic checks do not establish independence from all earlier parent-model data exposure. The fitted temperature is bound to the learned head; retraining invalidates it. Calibration can fail to transfer across populations or missing tasks, so report support and do not assume it improves every metric.

Finalize your raw/calibrated reporting protocol before accessing test metrics. Evaluate the selected raw checkpoint and, if fitted, its calibrated counterpart without using test results to launch another tuning cycle:

```bash
python -m paya.evaluate --config .temp/finetune.yaml --checkpoint runs/paya-custom-001/best.pt --split test --output runs/paya-custom-001/test-raw.json --device cuda
python -m paya.evaluate --config .temp/finetune.yaml --checkpoint runs/paya-custom-001/calibrated.pt --split test --output runs/paya-custom-001/test-calibrated.json --device cuda
```

The evaluation CLI writes pooled accuracy, NLL, Brier and ECE plus a `.predictions.jsonl` file. It does **not** automatically build the release's full per-task report. For one task's records, call `paya.classification.classification_metrics`; its rows must share an answer set, although candidate order may differ. Keep a task identifier in your analysis inputs and join predictions to it using a unique mapping ID. Do not mix different task vocabularies in one AUROC calculation. See [Results](results.md#metric-definitions) for aggregation conventions.

## 8. Export and verify

Export the finalized checkpoint to a new, empty directory:

```bash
python -m paya.export --config .temp/finetune.yaml --checkpoint runs/paya-custom-001/best.pt --output models/paya-custom-001
```

Substitute `calibrated.pt` only if that is your finalized artifact. Export includes the head, matching frozen assets, configuration and file manifest. Load it with `paya.inference.load_model("models/paya-custom-001")`, then verify its probabilities against the selected checkpoint on fixed real inputs before distribution. Keep the run's optimizer/recovery files separately; an inference export is not a resumable training run.

Record the parent and final checkpoint hashes, code revision, configuration, data-role hashes, task/class coverage and matching evaluation outputs in the new model card. If measuring latency, use that exact exported variant and document device, batch size, warm-up, cache use and timing boundaries. Fine-tuning results belong to the new checkpoint; the original release's figures and accuracy do not transfer to it.
