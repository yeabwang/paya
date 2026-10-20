"""Train the shared decision head over frozen native-rate signal embeddings."""

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F
import yaml

from .checkpoint import save_checkpoint, load_checkpoint
from .data import ChoiceDataset, assert_disjoint, make_batch
from .evaluate import collect, metrics
from .features import FeatureCache
from .model import PayaModel
from .tracking import open_run, run_identity, log_best


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
    cache = FeatureCache(config["feature_cache"]) if config.get("feature_cache") else None
    opt = config["optim"]
    epochs = opt["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Need at least one training epoch")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=opt["lr"], weight_decay=opt.get("weight_decay", 0.01))
    start, best, step, previous_run = 0, float("inf"), 0, None
    if resume:
        payload = load_checkpoint(resume, model)
        state = payload["training"]
        if not state or ({k: v for k, v in state["config"].items() if k != "wandb"}
                         != {k: v for k, v in config.items() if k != "wandb"}):
            raise ValueError("Resume requires the same model, data and optimizer configuration")
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"], state["best_nll"]
        step, previous_run = state["global_step"], state["wandb"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    history = []
    if start >= epochs:
        return model, history
    tracking = config.get("wandb", {})
    with open_run(config, destination, model, datasets, previous_run) as run:
        for epoch in range(start, epochs):
            started = perf_counter()
            total, count = 0.0, 0
            for rows in datasets["train"].batches(config.get("batch_size", 8), seed+epoch):
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
                optimizer.step()
                step += 1
                total += loss.item()*len(rows); count += len(rows)
                if run and step % tracking.get("log_every", 10) == 0:
                    logged = {"step": step, "epoch": epoch+1, "train/loss": loss.item(),
                              "train/answer_loss": loss.item(), "train/gradient_norm": float(gradient_norm),
                              "train/learning_rate": optimizer.param_groups[0]["lr"]}
                    run.log(logged)
            records = collect(model, datasets["selection"], config.get("batch_size", 8), cache)
            selection = metrics(records)
            improved = selection["nll"] < best
            if improved:
                best = selection["nll"]
            state = {"config": config, "epoch": epoch+1, "best_nll": best, "global_step": step,
                     "wandb": run_identity(run, tracking.get("mode", "disabled")),
                     "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                     "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "forbidden_calibration": {field: sorted({row[field] for name in ("train", "selection")
                                                             for row in datasets[name].rows})
                                               for field in ("source_id", "group_id", "recording_id")}}
            save_checkpoint(destination/"last.pt", model, state)
            if improved:
                save_checkpoint(destination/"best.pt", model, state)
            row = {"epoch": epoch+1, "phase": "decision_head",
                   "step": step, "train_loss": total/max(1, count), "selection": selection}
            with (destination/"history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
            history.append(row)
            if run:
                logged = {"epoch": epoch+1, "step": step, "phase": row["phase"],
                          "train/epoch_loss": row["train_loss"], "train/epoch_seconds": perf_counter()-started,
                          **{f"selection/{k}": selection[k] for k in ("accuracy", "nll", "brier", "ece_15", "count")}}
                run.log(logged)
                if improved:
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
