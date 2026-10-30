"""Train the shared decision head over frozen native-rate signal embeddings."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random

import torch
from torch.nn import functional as F
import yaml

from .checkpoint import save_checkpoint, load_checkpoint, learned_digest
from .data import ChoiceDataset, assert_disjoint, make_batch
from .evaluate import collect, metrics
from .features import open_feature_cache
from .model import PayaModel
from .tracking import open_run, run_identity, log_best
from .atomic import atomic_output
from .provenance import sha256, source_identity, runtime_identity, fingerprint
from .reporting import task_report


def choice_loss(logits, targets, mask, smoothing=0.0, reduction="mean"):
    if targets.dtype != torch.long or targets.shape != (len(logits),):
        raise ValueError("Targets must be integer candidate indices [B]")
    if ((targets < 0) | (targets >= logits.shape[1])).any():
        raise ValueError("Target out of range")
    if not mask.gather(1, targets[:, None]).all():
        raise ValueError("Cannot train on a padded candidate")
    if not 0 <= smoothing < 1 or reduction not in ("mean", "none"):
        raise ValueError("Invalid candidate loss smoothing or reduction")
    if smoothing == 0:
        return F.cross_entropy(logits, targets, reduction=reduction)
    logp = logits.log_softmax(-1)
    nll = -logp.gather(1, targets[:, None]).squeeze(1)
    uniform = -logp.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    values = (1 - smoothing) * nll + smoothing * uniform
    return values.mean() if reduction == "mean" else values


def train(config_path, run_dir, device="cpu", resume=None, initialize_from=None):
    """Start fresh, fine-tune head weights, or resume an identical interrupted run."""
    with ExitStack() as resources:
        return _train(config_path, run_dir, device, resume, initialize_from, resources)


def _train(config_path, run_dir, device, resume, initialize_from, resources):
    if resume and initialize_from:
        raise ValueError("Choose --resume or --initialize-from, not both")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config_digest = fingerprint(
        {key: value for key, value in config.items() if key != "wandb"}
    )
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()) and resume is None:
        raise FileExistsError("Use a fresh run directory or supply --resume")
    destination.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config.get("threads", 4))
    datasets = {
        name: ChoiceDataset(path)
        for name, path in config["data"].items()
        if name in ("train", "selection", "calibration", "test") and path
    }
    for dataset in datasets.values():
        resources.callback(dataset.close)
    if not {"train", "selection"} <= datasets.keys():
        raise ValueError("Training and independent selection datasets are required")
    assert_disjoint(*datasets.values())
    model = PayaModel.from_assets(
        config["assets"], config.get("preparation"), config.get("head"), device
    )
    initialization = None
    inherited_calibration_exclusions = []
    if initialize_from:
        payload = load_checkpoint(initialize_from, model)
        initialization = {
            "checkpoint_sha256": sha256(initialize_from),
            "learned_sha256": learned_digest(model),
        }
        inherited_calibration_exclusions = (
            (payload.get("training") or {})
            .get("forbidden_calibration", {})
            .get("datasets", [])
        )
        model.train()
    cache = open_feature_cache(config)
    if cache is not None:
        resources.callback(cache.close)
    opt = config["optim"]
    epochs = opt["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Need at least one training epoch")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=opt["lr"],
        weight_decay=opt.get("weight_decay", 0.01),
    )
    interval = opt.get("checkpoint_every_steps", 100)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("checkpoint_every_steps must be a positive integer")
    eval_interval = opt.get("eval_every_steps")
    max_steps = opt.get("max_steps")
    patience = opt.get("patience")
    for name, value in (
        ("eval_every_steps", eval_interval),
        ("max_steps", max_steps),
        ("patience", patience),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or null")
    bindings = {name: data.sha256 for name, data in datasets.items()}
    forbidden = {
        "datasets": [
            {"path": str(datasets[name].path), "sha256": datasets[name].sha256}
            for name in ("train", "selection")
        ]
    }
    forbidden["datasets"].extend(inherited_calibration_exclusions)
    start, best, step, previous_run = 0, float("inf"), 0, None
    next_batch, epoch_total, epoch_count, restored_state = 0, 0.0, 0, None
    last_eval, stale, stopped = 0, 0, False
    # Load lazy text assets before restoring RNG, so initialization cannot perturb resumed dropout.
    make_batch(model, [datasets["train"].rows[0]])
    if resume:
        payload = load_checkpoint(resume, model)
        if payload["source_sha256"] != source_identity():
            raise ValueError("Resume requires unchanged training source")
        if payload["runtime"] != runtime_identity():
            raise ValueError("Resume requires the same numerical runtime")
        state = payload["training"]
        if not state or (
            {k: v for k, v in state["config"].items() if k != "wandb"}
            != {k: v for k, v in config.items() if k != "wandb"}
        ):
            raise ValueError(
                "Resume requires the same model, data and optimizer configuration"
            )
        if state["config_sha256"] != config_digest:
            raise ValueError("Resume configuration checksum changed")
        optimizer.load_state_dict(state["optimizer"])
        if state.get("dataset_sha256") != bindings:
            raise ValueError("Resume dataset content differs from checkpoint")
        start, best = state["epoch"], state["best_nll"]
        step, previous_run = state["global_step"], state["wandb"]
        next_batch = state.get("next_batch", 0)
        epoch_total, epoch_count = (
            state.get("epoch_loss_sum", 0.0),
            state.get("epoch_count", 0),
        )
        restored_state = state
        last_eval, stale, stopped = state["last_eval"], state["stale"], state["stopped"]
        initialization = state["initialization"]
        forbidden = state["forbidden_calibration"]
    history = []
    completed_result = (
        restored_state.get("completed_epoch_result") if restored_state else None
    )
    last_batch_metrics = (
        restored_state.get("last_batch_metrics") if restored_state else None
    )
    history_path = destination / "history.jsonl"
    saved_history = []
    if history_path.exists():
        with history_path.open(encoding="utf-8") as source:
            saved_history = [json.loads(line) for line in source if line.strip()]
        saved_history = [row for row in saved_history if row["step"] <= step]
        with atomic_output(history_path, "w") as output:
            for row in saved_history:
                output.write(json.dumps(row) + "\n")
    steps_path = destination / "steps.jsonl"
    if restored_state and "metrics_bytes" in restored_state:
        committed_bytes = restored_state["metrics_bytes"]
        if not steps_path.exists() or steps_path.stat().st_size < committed_bytes:
            raise ValueError(
                "Resume requires the committed steps.jsonl beside the checkpoint"
            )
        with steps_path.open("r+b") as output:
            output.truncate(committed_bytes)
            output.flush()
            os.fsync(output.fileno())
    if start >= epochs or stopped or (max_steps is not None and step >= max_steps):
        return model, history
    tracking = config.get("wandb", {})
    with (
        open_run(config, destination, model, datasets, previous_run) as run,
        steps_path.open("a+b") as step_log,
    ):
        if restored_state:
            torch.set_rng_state(restored_state["torch_rng"])
            random.setstate(restored_state["python_rng"])
            if torch.cuda.is_available() and restored_state["cuda_rng"]:
                torch.cuda.set_rng_state_all(restored_state["cuda_rng"])
            if run:
                run.log(
                    {
                        "recovery/checkpoint_step": step,
                        "recovery/next_batch": next_batch,
                    }
                )
                if restored_state.get("metrics_bytes"):
                    with steps_path.open(encoding="utf-8") as source:
                        for line in source:
                            record = json.loads(line)
                            if (
                                record["step"] % tracking.get("log_every", 10) == 0
                                or record["step"] == step
                            ):
                                run.log({**record, "recovery/replayed": True})
                elif last_batch_metrics:
                    run.log({**last_batch_metrics, "recovery/replayed": True})
                for completed in saved_history:
                    run.log(
                        {
                            "step": completed["step"],
                            "epoch": completed["epoch"],
                            "recovery/replayed": True,
                            **{
                                f"selection/{key}": value
                                for key, value in completed["selection"].items()
                                if key != "reliability"
                            },
                        }
                    )

        def progress(epoch, cursor, total, count):
            step_log.flush()
            os.fsync(step_log.fileno())
            return {
                "config": config,
                "config_sha256": config_digest,
                "epoch": epoch,
                "next_batch": cursor,
                "metrics_bytes": step_log.tell(),
                "epoch_loss_sum": total,
                "epoch_count": count,
                "completed_epoch_result": completed_result,
                "last_batch_metrics": last_batch_metrics,
                "last_eval": last_eval,
                "stale": stale,
                "stopped": stopped,
                "initialization": initialization,
                "best_nll": best,
                "global_step": step,
                "dataset_sha256": bindings,
                "wandb": run_identity(run, tracking.get("mode", "disabled")),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                "forbidden_calibration": forbidden,
            }

        def select(epoch, cursor, total, count, completed_epoch=False):
            nonlocal best, stale, stopped, last_eval, completed_result, saved_history
            records = collect(
                model,
                datasets["selection"],
                config.get("batch_size", 8),
                cache,
                output_path=destination / "selection-predictions.jsonl",
            )
            selection = metrics(records)
            improved = selection["nll"] < best
            best = min(best, selection["nll"])
            stale = 0 if improved else stale + 1
            stopped = patience is not None and stale >= patience
            last_eval = step
            row = {
                "epoch": epoch + 1,
                "phase": "decision_head",
                "step": step,
                "train_loss": total / max(1, count),
                "selection": selection,
                "task_metrics": task_report(records),
            }
            completed_result = row
            saved_history = [
                previous for previous in saved_history if previous["step"] < step
            ] + [row]
            with atomic_output(history_path, "w") as log:
                for previous in saved_history:
                    log.write(json.dumps(previous, allow_nan=False) + "\n")
            state = (
                progress(epoch + 1, 0, 0.0, 0)
                if completed_epoch
                else progress(epoch, cursor, total, count)
            )
            if improved:
                save_checkpoint(destination / "best.pt", model, state)
            save_checkpoint(destination / "last.pt", model, state)
            history.append(row)
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "phase": row["phase"],
                        "train/epoch_loss": row["train_loss"],
                        **{
                            f"selection/{key}": selection[key]
                            for key in ("accuracy", "nll", "brier", "ece_15", "count")
                        },
                    }
                )
                if improved:
                    run.summary.update(
                        {
                            "best/epoch": epoch + 1,
                            "best/step": step,
                            "best/selection_nll": best,
                        }
                    )
                    if tracking.get("log_models", True):
                        log_best(run, model, destination, epoch + 1, best)
            print(json.dumps(row), flush=True)

        if not resume:
            save_checkpoint(destination / "last.pt", model, progress(0, 0, 0.0, 0))
        for epoch in range(start, epochs):
            for data in datasets.values():
                data.assert_unchanged()
            total, count = (epoch_total, epoch_count) if epoch == start else (0.0, 0)
            cursor = next_batch if epoch == start else 0
            for batch_index, rows in enumerate(
                datasets["train"].batches(
                    config.get("batch_size", 8), seed + epoch, start_batch=cursor
                ),
                start=cursor,
            ):
                batch = make_batch(model, rows)
                targets = torch.tensor(
                    [row["target"] for row in rows], device=model.device
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                result = model(batch, cache=cache)
                loss = choice_loss(
                    result["logits"],
                    targets,
                    batch.criteria_mask,
                    opt.get("label_smoothing", 0.0),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), 1.0
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Nonfinite training gradients")
                optimizer.step()
                step += 1
                total += loss.item() * len(rows)
                count += len(rows)
                cursor = batch_index + 1
                last_batch_metrics = {
                    "step": step,
                    "epoch": epoch + 1,
                    "train/loss": loss.item(),
                    "train/answer_loss": loss.item(),
                    "train/gradient_norm": float(gradient_norm),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }
                step_log.write(
                    (json.dumps(last_batch_metrics, allow_nan=False) + "\n").encode(
                        "utf-8"
                    )
                )
                if run and step % tracking.get("log_every", 10) == 0:
                    run.log(last_batch_metrics)
                if (eval_interval and step % eval_interval == 0) or (
                    max_steps is not None and step >= max_steps
                ):
                    select(epoch, cursor, total, count)
                if step % interval == 0:
                    save_checkpoint(
                        destination / "last.pt",
                        model,
                        progress(epoch, cursor, total, count),
                    )
                if stopped or (max_steps is not None and step >= max_steps):
                    break
            if step != last_eval:
                select(
                    epoch,
                    cursor,
                    total,
                    count,
                    completed_epoch=not stopped
                    and (max_steps is None or step < max_steps),
                )
            else:
                save_checkpoint(
                    destination / "last.pt",
                    model,
                    progress(epoch + 1, 0, 0.0, 0)
                    if not stopped and (max_steps is None or step < max_steps)
                    else progress(epoch, cursor, total, count),
                )
            if stopped or (max_steps is not None and step >= max_steps):
                break
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Fine-tune head weights with a fresh optimizer and temperature 1",
    )
    args = parser.parse_args()
    train(args.config, args.run_dir, args.device, args.resume, args.initialize_from)
