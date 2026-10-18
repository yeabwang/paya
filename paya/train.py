"""Train the shared head with deterministic selection and complete epoch resume."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from .assess import report
from .data import ChoiceDataset, ShapeSampler, loader, to_device
from .evaluate_cached import predict
from .features import load_cache
from .model import build_model
from .provenance import identity, sha256, source_sha256
from .text import IDENTITY, TextEncoder
from .tracking import Tracker


def seed_all(seed):
    """Seed Python, NumPy and PyTorch and disable CUDA TF32 for this training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def choice_loss(logits, labels, mask, smoothing=0.0):
    """Compute cross-entropy with optional smoothing over valid candidates only.

    logits and mask are [batch, choices]; labels is [batch]. Padded alternatives
    never contribute to smoothing. Reject padded targets and invalid smoothing.
    """
    if not 0 <= smoothing < 1 or not mask.gather(1, labels[:, None]).all():
        raise ValueError("Invalid smoothing or padded target")
    logp = logits.log_softmax(-1)
    hard = -logp.gather(1, labels[:, None]).squeeze(1)
    soft = -logp.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    return ((1 - smoothing) * hard + smoothing * soft).mean()


def save_checkpoint(
    path,
    model,
    config,
    provenance,
    optimizer,
    scheduler,
    epoch,
    step,
    best,
    stale,
    tracker_id,
    run_identity,
):
    """Atomically save head, optimizer, scheduler, selection and RNG state.

    The checkpoint also stores configuration and model/data/text provenance
    needed for exact continuation at a completed epoch boundary.
    """
    state = {
        "head": model.head.state_dict(),
        "temperature": float(model.temperature),
        "config": config,
        "identity": provenance,
        "text_identity": IDENTITY,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best": best,
        "stale": stale,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "tracker_id": tracker_id,
        "run_identity": run_identity,
    }
    temporary = Path(path).with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config_path, run_dir, device="cuda", resume=None, max_steps=None):
    """Train the shared head and select checkpoints on independent selection groups.

    Read a YAML config and write run metadata, metrics and checkpoints. Resume
    requires matching config/provenance and restores optimizer, schedule, and
    random states. max_steps writes a diagnostic partial epoch that cannot
    serve as an exact-resume checkpoint. The pretrained encoders stay frozen.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    directory = Path(run_dir)
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("threads", 4))
    seed_all(config["seed"])
    text = TextEncoder(config["data"]["text_cache"])
    text.encode(json.loads((Path(config["data"]["root"]) / "texts.json").read_text()))
    cache = load_cache(
        config["data"]["feature_cache"], config["data"]["root"], config["assets"]
    )
    options = {
        "root": config["data"]["root"],
        "text": text,
        "cache": cache,
        "exclude": config["data"].get("exclude", []),
        "families": config["data"].get("families"),
    }
    training = ChoiceDataset(
        split="train", augment=True, seed=config["seed"], **options
    )
    selection = ChoiceDataset(split="selection", **options)
    groups_train = sorted({r["group_id"] for r, _ in training.examples})
    groups_select = sorted({r["group_id"] for r, _ in selection.examples})
    if set(groups_train) & set(groups_select):
        raise ValueError("Training/selection groups overlap")
    provenance = identity(config["assets"])
    provenance_run = {
        "model": provenance,
        "data": sha256(Path(config["data"]["root"]) / "manifest.json"),
        "text_cache": sha256(config["data"]["text_cache"]),
        "training_code": source_sha256(__file__),
        "data_code": source_sha256(Path(__file__).with_name("data.py")),
    }
    if resume and (
        json.loads((directory / "provenance.json").read_text()) != provenance_run
        or yaml.safe_load((directory / "config.yaml").read_text()) != config
    ):
        raise ValueError(
            "Resume would change the saved run configuration or provenance"
        )
    (directory / "provenance.json").write_text(json.dumps(provenance_run, indent=2))
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    (directory / "groups.json").write_text(
        json.dumps({"train": groups_train, "selection": groups_select})
    )
    model = build_model(config["assets"], config).to(device)
    opt = config["optim"]
    batch_size = config["data"]["batch_size"]
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"]
    )
    steps_per_epoch = len(ShapeSampler(training, batch_size))
    total = max(1, steps_per_epoch * opt["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, int(total * 0.05)))
        * 0.5
        * (1 + math.cos(math.pi * min(step, total) / total)),
    )
    start, step, best, stale, tracker_id = 0, 0, (-1.0, -float("inf")), 0, None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            state["config"] != config
            or state["identity"] != provenance
            or state["run_identity"] != provenance_run
        ):
            raise ValueError("Resume configuration/model/data/text changed")
        model.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, step, best, stale, tracker_id = (
            state["epoch"],
            state["step"],
            tuple(state["best"]),
            state["stale"],
            state["tracker_id"],
        )
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    cpu_rng, python_rng = torch.get_rng_state(), random.getstate()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    tracker = Tracker(directory, config, tracker_id)
    torch.set_rng_state(cpu_rng)
    random.setstate(python_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    # Each family keeps its natural within-family label distribution; no inverse class weights.
    counts = Counter(
        (r["dataset_id"], q["family"], q["label"]) for r, q in training.examples
    )
    (directory / "training_distribution.json").write_text(
        json.dumps({"/".join(map(str, k)): v for k, v in counts.items()})
    )
    try:
        for epoch in range(start, opt["epochs"]):
            training.epoch = epoch
            model.train()
            loss_sum, n = 0.0, 0
            for batch in loader(training, batch_size, True):
                optimizer.zero_grad(set_to_none=True)
                logits = model(**to_device(batch["inputs"], device))
                loss = choice_loss(
                    logits,
                    batch["labels"].to(device),
                    batch["inputs"]["choice_mask"].to(device),
                    opt.get("label_smoothing", 0.0),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite choice loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                loss_sum += float(loss.detach()) * len(batch["labels"])
                n += len(batch["labels"])
                if step % 50 == 0:
                    tracker.log({"train/loss": loss_sum / n}, step)
                    print(
                        json.dumps(
                            {"epoch": epoch + 1, "step": step, "loss": loss_sum / n}
                        ),
                        flush=True,
                    )
                if max_steps and step >= max_steps:
                    # A diagnostic partial epoch is not an exact-resume checkpoint.
                    torch.save(
                        {
                            "head": model.head.state_dict(),
                            "config": config,
                            "identity": provenance,
                            "temperature": 1.0,
                            "diagnostic_only": True,
                        },
                        directory / "diagnostic.pt",
                    )
                    return {"diagnostic_steps": step, "loss": loss_sum / n}
            records = predict(model, selection, batch_size, device)
            assessment = report(records, bootstrap=0)
            m = assessment["native_mean"]
            score = ((m["accuracy"] + m["balanced_accuracy"]) / 2, -m["nll"])
            improved = score > best
            best = max(score, best)
            stale = 0 if improved else stale + 1
            row = {
                "epoch": epoch + 1,
                "step": step,
                "train_loss": loss_sum / n,
                "selection": assessment,
                "selection_rule": "native mean accuracy/balanced-accuracy, then lower NLL",
            }
            with (directory / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            tracker.log(
                {"selection/" + k: v for k, v in m.items() if v is not None}, step
            )
            save_checkpoint(
                directory / "latest.pt",
                model,
                config,
                provenance,
                optimizer,
                scheduler,
                epoch + 1,
                step,
                best,
                stale,
                tracker.id,
                provenance_run,
            )
            if improved:
                save_checkpoint(
                    directory / "best.pt",
                    model,
                    config,
                    provenance,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    step,
                    best,
                    stale,
                    tracker.id,
                    provenance_run,
                )
            print(
                json.dumps(
                    {"epoch": epoch + 1, "native_selection": m, "best": improved}
                ),
                flush=True,
            )
            if stale >= opt["patience"]:
                break
        return {
            "epochs": epoch + 1,
            "best_score": best,
            "checkpoint": str(directory / "best.pt"),
        }
    finally:
        tracker.finish()


def main():
    """Start, diagnose, or resume head training from a YAML configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-steps", type=int)
    a = p.parse_args()
    print(json.dumps(train(a.config, a.run_dir, a.device, a.resume, a.max_steps)))


if __name__ == "__main__":
    main()
