"""Train the shared decision head over frozen native-rate signal embeddings."""

import argparse
import json
import os
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F
import yaml

from .checkpoint import save_checkpoint, load_checkpoint
from .data import ChoiceDataset, assert_disjoint, make_batch
from .evaluate import collect, metrics
from .features import open_feature_cache
from .model import PayaModel
from .tracking import open_run, run_identity, log_best
from .atomic import atomic_output


def choice_loss(logits, targets, mask):
    if targets.dtype != torch.long or targets.shape != (len(logits),):
        raise ValueError("Targets must be integer candidate indices [B]")
    if ((targets < 0) | (targets >= logits.shape[1])).any():
        raise ValueError("Target out of range")
    if not mask.gather(1, targets[:, None]).all():
        raise ValueError("Cannot train on a padded candidate")
    return F.cross_entropy(logits, targets)


def train(config_path, run_dir, device="cpu", resume=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()) and resume is None:
        raise FileExistsError("Use a fresh run directory or supply --resume")
    destination.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 42)
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.get("threads", 4))
    datasets = {name: ChoiceDataset(path) for name, path in config["data"].items()
                if name in ("train", "selection", "calibration", "test") and path}
    if not {"train", "selection"} <= datasets.keys():
        raise ValueError("Training and independent selection datasets are required")
    assert_disjoint(*datasets.values())
    model = PayaModel.from_assets(config["assets"], config.get("preparation"), config.get("head"), device)
    cache = open_feature_cache(config)
    opt = config["optim"]
    epochs = opt["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Need at least one training epoch")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=opt["lr"], weight_decay=opt.get("weight_decay", 0.01))
    interval = opt.get("checkpoint_every_steps", 100)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("checkpoint_every_steps must be a positive integer")
    bindings = {name: data.sha256 for name, data in datasets.items()}
    forbidden = {"datasets": [{"path": str(datasets[name].path), "sha256": datasets[name].sha256}
                               for name in ("train", "selection")]}
    start, best, step, previous_run = 0, float("inf"), 0, None
    next_batch, epoch_total, epoch_count, restored_state = 0, 0.0, 0, None
    # Load lazy text assets before restoring RNG, so initialization cannot perturb resumed dropout.
    make_batch(model, [datasets["train"].rows[0]])
    if resume:
        payload = load_checkpoint(resume, model)
        state = payload["training"]
        if not state or ({k: v for k, v in state["config"].items() if k != "wandb"}
                         != {k: v for k, v in config.items() if k != "wandb"}):
            raise ValueError("Resume requires the same model, data and optimizer configuration")
        optimizer.load_state_dict(state["optimizer"])
        if state.get("dataset_sha256") != bindings:
            raise ValueError("Resume dataset content differs from checkpoint")
        start, best = state["epoch"], state["best_nll"]
        step, previous_run = state["global_step"], state["wandb"]
        next_batch = state.get("next_batch", 0)
        epoch_total, epoch_count = state.get("epoch_loss_sum", 0.0), state.get("epoch_count", 0)
        restored_state = state
    history = []
    completed_result = restored_state.get("completed_epoch_result") if restored_state else None
    last_batch_metrics = restored_state.get("last_batch_metrics") if restored_state else None
    history_path = destination/"history.jsonl"
    saved_history = []
    if history_path.exists():
        with history_path.open(encoding="utf-8") as source:
            saved_history = [json.loads(line) for line in source if line.strip()]
        saved_history = [row for row in saved_history if row["epoch"] <= start]
        with atomic_output(history_path,"w") as output:
            for row in saved_history:
                output.write(json.dumps(row)+"\n")
    steps_path = destination/"steps.jsonl"
    if restored_state and "metrics_bytes" in restored_state:
        committed_bytes = restored_state["metrics_bytes"]
        if not steps_path.exists() or steps_path.stat().st_size < committed_bytes:
            raise ValueError("Resume requires the committed steps.jsonl beside the checkpoint")
        with steps_path.open("r+b") as output:
            output.truncate(committed_bytes)
            output.flush()
            os.fsync(output.fileno())
    if start >= epochs:
        return model, history
    tracking = config.get("wandb", {})
    with open_run(config, destination, model, datasets, previous_run) as run, steps_path.open("a+b") as step_log:
        if restored_state:
            torch.set_rng_state(restored_state["torch_rng"])
            random.setstate(restored_state["python_rng"])
            if torch.cuda.is_available() and restored_state["cuda_rng"]:
                torch.cuda.set_rng_state_all(restored_state["cuda_rng"])
            if run:
                run.log({"recovery/checkpoint_step": step, "recovery/next_batch": next_batch})
                if restored_state.get("metrics_bytes"):
                    with steps_path.open(encoding="utf-8") as source:
                        for line in source:
                            record=json.loads(line)
                            if record["step"] % tracking.get("log_every",10)==0 or record["step"]==step:
                                run.log({**record,"recovery/replayed":True})
                elif last_batch_metrics:
                    run.log({**last_batch_metrics,"recovery/replayed":True})
                for completed in saved_history:
                    run.log({"step":completed["step"],"epoch":completed["epoch"],"recovery/replayed":True,
                        **{f"selection/{key}":value for key,value in completed["selection"].items() if key!='reliability'}})

        def progress(epoch, cursor, total, count):
            step_log.flush()
            os.fsync(step_log.fileno())
            return {"config": config, "epoch": epoch, "next_batch": cursor,
                    "metrics_bytes":step_log.tell(),
                    "epoch_loss_sum": total, "epoch_count": count,
                    "completed_epoch_result": completed_result,
                    "last_batch_metrics": last_batch_metrics,
                    "best_nll": best, "global_step": step, "dataset_sha256": bindings,
                    "wandb": run_identity(run, tracking.get("mode", "disabled")),
                    "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "forbidden_calibration": forbidden}

        if not resume:
            save_checkpoint(destination/"last.pt", model, progress(0, 0, 0.0, 0))
        for epoch in range(start, epochs):
            started = perf_counter()
            for data in datasets.values():
                data.assert_unchanged()
            total, count = (epoch_total, epoch_count) if epoch == start else (0.0, 0)
            cursor = next_batch if epoch == start else 0
            for batch_index, rows in enumerate(datasets["train"].batches(
                    config.get("batch_size", 8), seed+epoch, start_batch=cursor), start=cursor):
                batch = make_batch(model, rows)
                targets = torch.tensor([row["target"] for row in rows], device=model.device)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                result = model(batch, cache=cache)
                loss = choice_loss(result["logits"], targets, batch.criteria_mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Nonfinite training gradients")
                optimizer.step()
                step += 1
                total += loss.item()*len(rows); count += len(rows)
                cursor = batch_index+1
                last_batch_metrics = {"step": step, "epoch": epoch+1, "train/loss": loss.item(),
                    "train/answer_loss": loss.item(), "train/gradient_norm": float(gradient_norm),
                    "train/learning_rate": optimizer.param_groups[0]["lr"]}
                step_log.write((json.dumps(last_batch_metrics,allow_nan=False)+"\n").encode("utf-8"))
                if run and step % tracking.get("log_every", 10) == 0:
                    run.log(last_batch_metrics)
                if step % interval == 0:
                    save_checkpoint(destination/"last.pt", model, progress(epoch, cursor, total, count))
            records = collect(model, datasets["selection"], config.get("batch_size", 8), cache,
                              output_path=destination/"selection-predictions.jsonl")
            selection = metrics(records)
            improved = selection["nll"] < best
            if improved:
                best = selection["nll"]
            row = {"epoch": epoch+1, "phase": "decision_head",
                   "step": step, "train_loss": total/max(1, count), "selection": selection}
            completed_result = row
            state = progress(epoch+1, 0, 0.0, 0)
            if improved:
                save_checkpoint(destination/"best.pt", model, state)
            saved_history = [previous for previous in saved_history if previous["epoch"] < epoch+1]+[row]
            with atomic_output(history_path,"w") as log:
                for previous in saved_history:
                    log.write(json.dumps(previous)+"\n")
            save_checkpoint(destination/"last.pt", model, state)
            history.append(row)
            if run:
                logged = {"epoch": epoch+1, "step": step, "phase": row["phase"],
                          "train/epoch_loss": row["train_loss"], "train/epoch_seconds": perf_counter()-started,
                          **{f"selection/{k}": selection[k] for k in ("accuracy", "nll", "brier", "ece_15", "count")}}
                run.log(logged)
                if improved:
                    run.summary.update({"best/epoch":epoch+1,"best/selection_nll":best})
                    if tracking.get("log_models",True):
                        log_best(run, model, destination, epoch+1, best)
            print(json.dumps(row), flush=True)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paya.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.config, args.run_dir, args.device, args.resume)
