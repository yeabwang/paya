"""Train the new shared head and two-expert gate; never fine-tune the encoders."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
import yaml

from .checkpoint import save_checkpoint, load_checkpoint
from .data import ChoiceDataset, assert_disjoint, make_batch
from .evaluate import collect, metrics
from .features import FeatureCache
from .model import PayaModel, DecisionBatch


def choice_loss(logits, targets, mask):
    if targets.dtype != torch.long or targets.shape != (len(logits),):
        raise ValueError("Targets must be integer candidate indices [B]")
    if ((targets < 0) | (targets >= logits.shape[1])).any():
        raise ValueError("Target out of range")
    if not mask.gather(1, targets[:, None]).all():
        raise ValueError("Cannot train on a padded candidate")
    return F.cross_entropy(logits, targets)


def subset(batch, indices):
    ids = indices.tolist()
    return DecisionBatch([batch.states[i] for i in ids], batch.question[indices], batch.criteria[indices],
                         batch.ending[indices], batch.criteria_mask[indices], [batch.descriptions[i] for i in ids])


def expert_losses(model, batch, targets, cache=None, gradients=False):
    """Compare only the two fixed experts for training-time routing supervision."""
    compatible = torch.stack([s.compatible for s in batch.states]).to(model.device)
    losses = batch.question.new_full((len(targets), 2), torch.inf)
    for expert in range(2):
        indices = compatible[:, expert].nonzero().flatten()
        if indices.numel():
            with torch.set_grad_enabled(gradients):
                current = subset(batch, indices)
                routes = torch.full((len(indices),), expert, device=model.device, dtype=torch.long)
                logits = model(current, routes=routes, cache=cache)["logits"]
                losses[indices, expert] = F.cross_entropy(logits, targets[indices], reduction="none")
    return losses, compatible


def routing_targets(losses, compatible, costs=(1.0, 1.0), cost_weight=0.0, temperature=1.0):
    if temperature <= 0 or cost_weight < 0 or len(costs) != 2 or any(c < 0 for c in costs):
        raise ValueError("Invalid route-target settings")
    if losses.shape != compatible.shape or not compatible.any(-1).all():
        raise ValueError("Invalid route eligibility")
    scores = -(losses + cost_weight*losses.new_tensor(costs)) / temperature
    return scores.masked_fill(~compatible, -torch.inf).softmax(-1).detach()


def train(config_path, run_dir, device="cpu", resume=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()) and resume is None:
        raise FileExistsError("Use a fresh run directory for the retrained architecture")
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
    epochs, warmup = opt["epochs"], opt.get("warmup_epochs", 1)
    if not isinstance(epochs, int) or epochs < 1 or not 0 <= warmup < epochs:
        raise ValueError("Need at least one routed training epoch after head warmup")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=opt["lr"], weight_decay=opt.get("weight_decay", 0.01))
    start, best = 0, float("inf")
    if resume:
        payload = load_checkpoint(resume, model)
        state = payload["training"]
        if not state or state["config"] != config:
            raise ValueError("Resume requires an exact new-contract training configuration")
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"], state["best_nll"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    history = []
    for epoch in range(start, epochs):
        total, count = 0.0, 0
        for rows in datasets["train"].batches(config.get("batch_size", 8), seed+epoch):
            batch = make_batch(model, rows)
            targets = torch.tensor([row["target"] for row in rows], device=model.device)
            optimizer.zero_grad(set_to_none=True)
            if epoch < warmup:
                model.train()
                losses, compatible = expert_losses(model, batch, targets, cache, gradients=True)
                # Every eligible expert trains the same head during warmup.
                loss = torch.stack([losses[compatible[:, expert], expert].mean()
                                    for expert in range(2) if compatible[:, expert].any()]).mean()
            else:
                model.eval()
                with torch.no_grad():
                    losses, compatible = expert_losses(model, batch, targets, cache)
                    teacher = routing_targets(losses, compatible, opt.get("expert_costs", [1.0, 1.0]),
                                              opt.get("cost_weight", 0.0), opt.get("route_temperature", 1.0))
                model.train()
                result = model(batch, cache=cache)
                decision = choice_loss(result["logits"], targets, batch.criteria_mask)
                log_routes = result["router_logits"].log_softmax(-1).masked_fill(~compatible, 0)
                gate_loss = -(teacher*log_routes).sum(-1).mean()
                loss = decision + opt.get("router_loss_weight", 1.0)*gate_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            optimizer.step()
            total += loss.item()*len(rows); count += len(rows)
        records = collect(model, datasets["selection"], config.get("batch_size", 8), cache)
        selection = metrics(records)
        # Warmup has not trained the gate. Only the actual routed computation can
        # be selected as the final architecture's best checkpoint.
        improved = epoch >= warmup and selection["nll"] < best
        if improved:
            best = selection["nll"]
        state = {"config": config, "epoch": epoch+1, "best_nll": best,
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "forbidden_calibration": {field: sorted({row[field] for name in ("train", "selection")
                                                         for row in datasets[name].rows})
                                           for field in ("source_id", "group_id", "recording_id")}}
        save_checkpoint(destination/"last.pt", model, state)
        if improved:
            save_checkpoint(destination/"best.pt", model, state)
        row = {"epoch": epoch+1, "phase": "head_warmup" if epoch < warmup else "routed",
               "train_loss": total/max(1, count), "selection": selection}
        with (destination/"history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        history.append(row)
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
